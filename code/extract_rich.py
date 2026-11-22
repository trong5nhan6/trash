"""Method M5: richer frozen features + flip test-time averaging.

ViT (DINOv2): concat CLS token of the last 4 blocks + mean patch token of the last block
              (the linear-evaluation recipe of the DINOv2 paper).
ViT without CLS (SigLIP): pooled (MAP) output + mean patch token of the last 4 blocks.
ResNet: global-average-pooled layer3 + layer4.
Every image is encoded as original and horizontally flipped; the two vectors are averaged.
Output: outputs/feats_rich/<backbone>.npz (feats, row_id) aligned like outputs/feats/.

Usage: python extract_rich.py --root <data> --out outputs --backbones dinov2_b siglip_b16 resnet50
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from torch import nn
from torch.utils.data import DataLoader
from extract import BACKBONES, IMG_SIZE, ImageList


def build(name, pretrained=True):
    kw = {"pretrained": pretrained, "num_classes": 0}
    if "dinov2" in name: kw["img_size"] = IMG_SIZE
    model = timm.create_model(BACKBONES[name], **kw).eval()
    cfg = timm.data.resolve_model_data_config(model)
    cfg.update(input_size=(3, IMG_SIZE, IMG_SIZE), crop_pct=1.0)
    return model, timm.data.create_transform(**cfg, is_training=False)


class Rich(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m, self.cache = model, []
        if hasattr(model, "blocks") and hasattr(model, "num_prefix_tokens"):
            self.kind, layers = "vit", list(model.blocks)[-4:]
        elif hasattr(model, "layer4"):
            self.kind, layers = "resnet", [model.layer3, model.layer4]
        else:
            self.kind, layers = "plain", []
        for l in layers:
            l.register_forward_hook(lambda mod, inp, out: self.cache.append(out))

    def encode(self, x):
        self.cache = []
        pooled = self.m(x)
        if self.kind == "vit":
            npt = self.m.num_prefix_tokens
            toks = [self.m.norm(t) for t in self.cache]
            if npt > 0:
                parts = [t[:, 0] for t in toks] + [toks[-1][:, npt:].mean(1)]
            else:
                parts = [pooled] + [t.mean(1) for t in toks]
        elif self.kind == "resnet":
            parts = [o.mean((2, 3)) for o in self.cache]
        else:
            parts = [pooled]
        return torch.cat(parts, 1)

    def forward(self, x):
        return (self.encode(x) + self.encode(torch.flip(x, dims=[3]))) / 2


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--backbones", nargs="+", default=["dinov2_b", "siglip_b16", "resnet50"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images only")
    ap.add_argument("--no-pretrained", action="store_true", help="debug: random weights")
    args = ap.parse_args()
    out = Path(args.out); (out / "feats_rich").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(out / "manifest.csv")
    sel = man[(man.keep == 1) & (man.core == 1)]
    if args.limit: sel = sel.iloc[:args.limit]
    row_id, paths = sel.index.to_numpy(), sel.path.tolist()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for name in args.backbones:
        f_out = out / "feats_rich" / f"{name}.npz"
        if f_out.exists():
            print(f"[skip] {name}"); continue
        t0 = time.time()
        model, tf = build(name, pretrained=not args.no_pretrained)
        enc = Rich(model).to(dev).eval()
        dl = DataLoader(ImageList(args.root, paths, tf), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=(dev == "cuda"))
        feats = []
        for x in dl:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                feats.append(enc(x.to(dev, non_blocking=True)).float().cpu())
        feats = torch.cat(feats).numpy().astype(np.float16)
        np.savez(f_out, feats=feats, row_id=row_id)
        print(f"[done] {name} ({enc.kind}): {feats.shape} in {time.time() - t0:.0f}s", flush=True)
        del enc, model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
