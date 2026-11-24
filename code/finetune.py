"""Method M6 (+ M6+CG): LP-FT (Kumar et al., ICLR'22), optionally followed by CG-SHOT-LP on the
fine-tuned features.

1. LP : logistic-regression probe on frozen features (same as B0), converted to a torch head.
2. FT : starting from the LP head, fine-tune
          ViT (DINOv2, CLIP, SigLIP)        -> LoRA (r=8) on attention qkv + proj
          CNN (ResNet / ConvNeXt / EffNet)  -> last stage (BatchNorm statistics frozen)
        together with the head for --epochs epochs; best epoch chosen on *source* val macro-F1.
3. CG : (with --cg) extract features of the fine-tuned backbone, fit a fresh source probe on them and run
        CG-SHOT-LP (methods_data.cg_predict) on every target test split. Label-free on targets.
Variants: lp, ft, ft_probe, ft_cg_hard, ft_cg_soft, ft_shot_lprop_auto, ft_t3a_ens, ft_cg_hard_t3a
(--wise adds the WiSE alpha=0.5 interpolation, reported earlier as harmful for CNNs.)

Usage: python finetune.py --root /tmp/small --out outputs --backbone dinov2_b --seeds 0 --cg --tag lpftcg
Output: outputs/results_methods/<tag>/parts/<backbone>_seed<k>_<protocol>.csv (+ _pred.csv.gz)
"""
import argparse, copy, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T
from scipy.special import softmax
from common import CORE, CORE_TO_ID, DATASET_NAMES, make_split
from extract import BACKBONES, IMG_SIZE
from probe import fit_probe, load_kept
from run_methods import PROTOCOLS, metrics


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scaling, self.mult = base, alpha / r, 1.0
        for p in base.parameters(): p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * (self.scaling * self.mult)


def cnn_last_stage(bb):
    """Modules fine-tuned for CNN backbones (the last stage + its output norm)."""
    if hasattr(bb, "layer4"):                                   # ResNet
        return [bb.layer4]
    if hasattr(bb, "stages"):                                   # ConvNeXt
        return [bb.stages[-1]] + ([bb.head] if hasattr(bb, "head") else [])
    if hasattr(bb, "conv_head"):                                # EfficientNet
        return [bb.blocks[-1], bb.conv_head, bb.bn2]
    raise ValueError(f"unsupported CNN backbone {type(bb).__name__}")


class Net(nn.Module):
    def __init__(self, name, W, b, pretrained=True):
        super().__init__()
        kw = {"pretrained": pretrained, "num_classes": 0}
        if "dinov2" in name: kw["img_size"] = IMG_SIZE
        self.backbone = timm.create_model(BACKBONES[name], **kw)
        for p in self.backbone.parameters(): p.requires_grad_(False)
        self.is_vit = hasattr(self.backbone, "num_prefix_tokens") and hasattr(self.backbone, "blocks")
        if self.is_vit:
            self.lora = []
            for blk in self.backbone.blocks:
                blk.attn.qkv = LoRALinear(blk.attn.qkv); blk.attn.proj = LoRALinear(blk.attn.proj)
                self.lora += [blk.attn.qkv, blk.attn.proj]
        else:
            self.stage = nn.ModuleList(cnn_last_stage(self.backbone))
            for p in self.stage.parameters(): p.requires_grad_(True)
            self.stage_init = copy.deepcopy(self.stage.state_dict())
        self.head = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(W)); self.head.bias.copy_(torch.tensor(b))
        self.head_init = copy.deepcopy(self.head.state_dict())

    def train(self, mode=True):
        super().train(mode)
        for m in self.backbone.modules():                     # keep BatchNorm running stats frozen
            if isinstance(m, nn.BatchNorm2d): m.eval()
        return self

    def forward(self, x):
        return self.head(self.backbone(x))

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_alpha(self, alpha, ft_state):
        """alpha=0 -> LP model, 1 -> fine-tuned; linear interpolation of fine-tuned parameters."""
        self.load_state_dict(ft_state)
        with torch.no_grad():
            for k, v in self.head.state_dict().items():
                v.copy_((1 - alpha) * self.head_init[k].to(v.device) + alpha * v)
            if self.is_vit:
                for m in self.lora: m.mult = alpha
            else:
                for k, v in self.stage.state_dict().items():
                    if v.dtype.is_floating_point:
                        v.copy_((1 - alpha) * self.stage_init[k].to(v.device) + alpha * v)


class DS(Dataset):
    def __init__(self, root, paths, labels, tf):
        self.root, self.paths, self.labels, self.tf = Path(root), paths, labels, tf

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            return self.tf(im.convert("RGB")), self.labels[i]


@torch.no_grad()
def run(model, dl, dev, feats=False):
    model.eval(); out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            o = model.backbone(x.to(dev)) if feats else model(x.to(dev))
        out.append(o.float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="image root (pre-resized cache recommended)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--protocols", nargs="*", type=int, default=None, help="indices into PROTOCOLS")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cg", action="store_true", help="run CG-SHOT-LP on the fine-tuned features")
    ap.add_argument("--wise", action="store_true", help="also report WiSE alpha=0.5")
    ap.add_argument("--tag", default="lpft", help="results_methods/<tag>/parts")
    ap.add_argument("--debug-n", type=int, default=None, help="debug: subsample each split")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); parts = out / "results_methods" / args.tag / "parts"; parts.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(out / "manifest.csv")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, row_id = load_kept(np.load(out / "feats" / f"{args.backbone}.npz"), man)
    m = man.loc[row_id]
    ds, y, paths = m.dataset.to_numpy(), m.label.map(CORE_TO_ID).to_numpy(), m.path.to_numpy()

    tmp = timm.create_model(BACKBONES[args.backbone], pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(tmp); del tmp
    norm = T.Normalize(cfg["mean"], cfg["std"])
    tf_eval = T.Compose([T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
                         T.CenterCrop(IMG_SIZE), T.ToTensor(), norm])
    tf_train = T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])

    def sub(idx):   # debug only: random (class-mixed) subsample
        return np.sort(np.random.default_rng(0).permutation(idx)[:args.debug_n]) if args.debug_n else idx

    def loader(idx, tf, shuffle):
        return DataLoader(DS(args.root, paths[idx], y[idx], tf), batch_size=args.bs, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=dev == "cuda", drop_last=shuffle and len(idx) > args.bs)

    proto_ids = args.protocols if args.protocols is not None else range(len(PROTOCOLS))
    for seed in args.seeds:
        split = make_split(ds, y, seed=seed)
        for pi in proto_ids:
            proto, srcs = PROTOCOLS[pi]
            tag = f"{args.backbone}_seed{seed}_{proto}-{'+'.join(srcs)}"
            if (parts / f"{tag}.csv").exists():
                print(f"[skip] {tag}"); continue
            t0 = time.time(); torch.manual_seed(seed)
            src = np.isin(ds, srcs)
            tr = sub(np.where(src & (split == "train"))[0]); va = sub(np.where(src & (split == "val"))[0])
            targets = DATASET_NAMES if proto == "single" else [d for d in DATASET_NAMES if d not in srcs]
            tests = {t: sub(np.where((ds == t) & (split == "test"))[0]) for t in targets}
            # 1) LP head from frozen features, converted from standardised to raw feature space
            sc, clf, C, _ = fit_probe(X[tr], y[tr], X[va], y[va], seed)
            W = clf.coef_ / sc.scale_[None]; b = clf.intercept_ - W @ sc.mean_
            model = Net(args.backbone, W.astype(np.float32), b.astype(np.float32),
                        pretrained=not args.no_pretrained).to(dev)
            # 2) fine-tune
            dl_tr, dl_va = loader(tr, tf_train, True), loader(va, tf_eval, False)
            opt = torch.optim.AdamW(model.trainable(), lr=args.lr, weight_decay=0.0)
            steps = args.epochs * max(1, len(dl_tr)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
            best = (-1, None)
            for ep in range(args.epochs):
                model.train()
                for xb, yb in dl_tr:
                    with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                        loss = nn.functional.cross_entropy(model(xb.to(dev)), yb.to(dev))
                    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update(); sched.step()
                f1 = metrics(softmax(run(model, dl_va, dev), 1), y[va])[0]["macro_f1"]
                if f1 > best[0]: best = (f1, copy.deepcopy(model.state_dict()))
                print(f"  {tag} ep{ep} loss={loss.item():.3f} val_f1={f1:.1f}", flush=True)
            ft_state = best[1]
            # 3) evaluate
            rows, preds = [], []

            def rec(var, tgt, prob):
                te = tests[tgt]
                mt, pred = metrics(prob, y[te])
                rows.append({"method": args.tag, "variant": var, "backbone": args.backbone, "seed": seed,
                             "protocol": proto, "train_on": "+".join(srcs), "test_on": tgt,
                             "in_domain": int(tgt in srcs), **mt})
                preds.append(pd.DataFrame({"variant": var, "backbone": args.backbone, "protocol": proto,
                                           "train_on": "+".join(srcs), "test_on": tgt, "row_id": row_id[te],
                                           "y_true": np.array(CORE)[y[te]], "y_pred": np.array(CORE)[pred],
                                           "conf": prob.max(1).round(4)}))

            alphas = [("lp", 0.0)] + ([("wise", 0.5)] if args.wise else []) + [("ft", 1.0)]
            for var, alpha in alphas:
                model.set_alpha(alpha, ft_state)
                for tgt, te in tests.items():
                    rec(var, tgt, softmax(run(model, loader(te, tf_eval, False), dev), 1))
            if args.cg:   # model is now at alpha=1 (fine-tuned)
                from methods_data import cg_predict, CG_VARIANTS
                F_tr = run(model, loader(tr, tf_eval, False), dev, feats=True)
                F_va = run(model, loader(va, tf_eval, False), dev, feats=True)
                F_te = {t: run(model, loader(te, tf_eval, False), dev, feats=True) for t, te in tests.items()}
                sc2, clf2, _, _ = fit_probe(F_tr, y[tr], F_va, y[va], seed)
                for tgt in tests:
                    rec("ft_probe", tgt, clf2.predict_proba(sc2.transform(F_te[tgt])))
                out_cg, tau, kept = cg_predict(sc2, clf2, F_va, y[va], F_te)
                for var in CG_VARIANTS:
                    for tgt in tests:
                        rec(f"ft_{var}", tgt, out_cg[var][tgt])
                        rows[-1]["tau"], rows[-1]["kept_frac"] = tau, kept[tgt]
            pd.concat(preds).to_csv(parts / f"{tag}_pred.csv.gz", index=False, compression="gzip")
            pd.DataFrame(rows).to_csv(parts / f"{tag}.csv", index=False)
            r = pd.DataFrame(rows)
            show = [v for v in ["lp", "ft", "ft_cg_hard", "ft_cg_soft"] if v in set(r.variant)]
            print(f"[done] {tag} {time.time() - t0:.0f}s | OOD F1 " +
                  " ".join(f"{v}={r[(r.variant == v) & (r.in_domain == 0)].macro_f1.mean():.1f}" for v in show),
                  flush=True)
            del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
