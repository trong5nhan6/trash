"""Method M6: LP-FT (Kumar et al., ICLR'22) + WiSE weight interpolation (Wortsman et al., CVPR'22).

1. LP : logistic-regression probe on frozen features (same as B0), converted to a torch head.
2. FT : starting from the LP head, fine-tune  ViT -> LoRA (r=8) on attention qkv+proj,
                                             ResNet -> layer4 (BatchNorm stats frozen),
        together with the head, for --epochs epochs; best epoch chosen on *source* val macro-F1.
3. WiSE: interpolate every fine-tuned parameter between LP (alpha=0) and FT (alpha=1).
Variants reported: lp (alpha 0), wise (alpha 0.5, fixed a priori), ft (alpha 1).

Usage: python finetune.py --root /tmp/small --out outputs --backbone dinov2_b --seeds 0
Output: outputs/results_methods/lpft/parts/<backbone>_seed<k>_<protocol>.csv (+ _pred.csv.gz)
        -> merged by summarize.py --method lpft
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


class Net(nn.Module):
    def __init__(self, name, W, b, pretrained=True):
        super().__init__()
        kw = {"pretrained": pretrained, "num_classes": 0}
        if "dinov2" in name: kw["img_size"] = IMG_SIZE
        self.backbone = timm.create_model(BACKBONES[name], **kw)
        for p in self.backbone.parameters(): p.requires_grad_(False)
        self.is_vit = hasattr(self.backbone, "blocks")
        if self.is_vit:
            self.lora = []
            for blk in self.backbone.blocks:
                blk.attn.qkv = LoRALinear(blk.attn.qkv); blk.attn.proj = LoRALinear(blk.attn.proj)
                self.lora += [blk.attn.qkv, blk.attn.proj]
        else:
            for p in self.backbone.layer4.parameters(): p.requires_grad_(True)
            self.layer4_init = copy.deepcopy(self.backbone.layer4.state_dict())
        self.head = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(W)); self.head.bias.copy_(torch.tensor(b))
        self.head_init = copy.deepcopy(self.head.state_dict())

    def train(self, mode=True):
        super().train(mode)
        if not self.is_vit:            # keep BatchNorm running stats frozen
            for m in self.backbone.modules():
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
                sd = self.backbone.layer4.state_dict()
                for k, v in sd.items():
                    if v.dtype.is_floating_point:
                        v.copy_((1 - alpha) * self.layer4_init[k].to(v.device) + alpha * v)


class DS(Dataset):
    def __init__(self, root, paths, labels, tf):
        self.root, self.paths, self.labels, self.tf = Path(root), paths, labels, tf

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            return self.tf(im.convert("RGB")), self.labels[i]


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(x.to(dev)).float().cpu())
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
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--debug-n", type=int, default=None, help="debug: subsample each split")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); parts = out / "results_methods" / "lpft" / "parts"; parts.mkdir(parents=True, exist_ok=True)
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

    def loader(idx, tf, shuffle):
        if args.debug_n: idx = idx[:args.debug_n]
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
            tr, va = np.where(src & (split == "train"))[0], np.where(src & (split == "val"))[0]
            # 1) LP head from frozen features, converted from standardised to raw feature space
            sc, clf, C, _ = fit_probe(X[tr], y[tr], X[va], y[va], seed)
            W = clf.coef_ / sc.scale_[None]; b = clf.intercept_ - W @ sc.mean_
            model = Net(args.backbone, W.astype(np.float32), b.astype(np.float32),
                        pretrained=not args.no_pretrained).to(dev)
            # 2) fine-tune
            dl_tr, dl_va = loader(tr, tf_train, True), loader(va, tf_eval, False)
            opt = torch.optim.AdamW(model.trainable(), lr=args.lr, weight_decay=0.0)
            steps = args.epochs * max(1, len(dl_tr)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
            yva = y[va][:args.debug_n] if args.debug_n else y[va]
            best = (-1, None)
            for ep in range(args.epochs):
                model.train()
                for xb, yb in dl_tr:
                    with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                        loss = nn.functional.cross_entropy(model(xb.to(dev)), yb.to(dev))
                    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update(); sched.step()
                f1 = metrics(softmax(predict(model, dl_va, dev), 1), yva)[0]["macro_f1"]
                if f1 > best[0]: best = (f1, copy.deepcopy(model.state_dict()))
                print(f"  {tag} ep{ep} loss={loss.item():.3f} val_f1={f1:.1f}", flush=True)
            ft_state = best[1]
            # 3) evaluate LP / WiSE / FT on every target test split
            targets = DATASET_NAMES if proto == "single" else [d for d in DATASET_NAMES if d not in srcs]
            rows, preds = [], []
            for var, alpha in [("lp", 0.0), ("wise", args.alpha), ("ft", 1.0)]:
                model.set_alpha(alpha, ft_state)
                for tgt in targets:
                    te = np.where((ds == tgt) & (split == "test"))[0]
                    if args.debug_n: te = te[:args.debug_n]
                    prob = softmax(predict(model, loader(te, tf_eval, False), dev), 1)
                    mt, pred = metrics(prob, y[te])
                    rows.append({"method": "lpft", "variant": var, "backbone": args.backbone, "seed": seed,
                                 "protocol": proto, "train_on": "+".join(srcs), "test_on": tgt,
                                 "in_domain": int(tgt in srcs), **mt})
                    preds.append(pd.DataFrame({"variant": var, "backbone": args.backbone, "protocol": proto,
                                               "train_on": "+".join(srcs), "test_on": tgt, "row_id": row_id[te],
                                               "y_true": np.array(CORE)[y[te]], "y_pred": np.array(CORE)[pred],
                                               "conf": prob.max(1).round(4)}))
            pd.concat(preds).to_csv(parts / f"{tag}_pred.csv.gz", index=False, compression="gzip")
            pd.DataFrame(rows).to_csv(parts / f"{tag}.csv", index=False)
            r = pd.DataFrame(rows)
            print(f"[done] {tag} {time.time() - t0:.0f}s | OOD F1 " +
                  " ".join(f"{v}={r[(r.variant == v) & (r.in_domain == 0)].macro_f1.mean():.1f}"
                           for v in ["lp", "wise", "ft"]), flush=True)
            del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
