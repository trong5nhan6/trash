"""Step 2: extract frozen features (no training) for every kept core image, one file per backbone.

Usage:  python extract.py --root <data root> --out outputs [--backbones resnet50 dinov2_b ...]
Output: outputs/feats/<backbone>.npz  with  feats [N, D] float16  and  row_id [N] (row index in manifest.csv)
Already-extracted backbones are skipped, so the script can be re-run after a Kaggle session dies.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# All models come from timm so preprocessing/pooling is handled uniformly.
BACKBONES = {
    "resnet50":   "resnet50.tv_in1k",                  # supervised CNN (classic torchvision weights)
    "effnet_b0":  "efficientnet_b0.ra_in1k",           # supervised CNN, light
    "convnext_t": "convnext_tiny.fb_in1k",             # supervised modern CNN
    "dinov2_b":   "vit_base_patch14_dinov2.lvd142m",   # self-supervised ViT
    "dinov2_l":   "vit_large_patch14_dinov2.lvd142m",
    "clip_b16":   "vit_base_patch16_clip_224.openai",  # image-text (image tower)
    "siglip_b16": "vit_base_patch16_siglip_224.webli",
}
IMG_SIZE = 224  # same input resolution for every backbone (fair comparison)


class ImageList(Dataset):
    def __init__(self, root, paths, tf):
        self.root, self.paths, self.tf = Path(root), paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def build(name):
    kw = {"pretrained": True, "num_classes": 0}           # num_classes=0 -> pooled features
    if "dinov2" in name:
        kw["img_size"] = IMG_SIZE                          # resample pos-embed 518 -> 224
    model = timm.create_model(BACKBONES[name], **kw).eval()
    cfg = timm.data.resolve_model_data_config(model)
    cfg.update(input_size=(3, IMG_SIZE, IMG_SIZE), crop_pct=1.0)  # keep whole object, no border crop
    tf = timm.data.create_transform(**cfg, is_training=False)
    return model, tf


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", default=None,
                    help="CSV with columns path,row_id,k (e.g. CLAD composites); paths relative to --root")
    ap.add_argument("--tag", default=None, help="with --list: features go to outputs/feats_<tag>/")
    args = ap.parse_args()

    out = Path(args.out)
    extra = {}
    if args.list:
        lst = pd.read_csv(args.list)
        row_id, paths, extra = lst.row_id.to_numpy(), lst.path.tolist(), {"k": lst.k.to_numpy()}
        feat_dir = out / f"feats_{args.tag}"
    else:
        man = pd.read_csv(out / "manifest.csv")
        sel = man[(man.keep == 1) & (man.core == 1)]
        row_id, paths = sel.index.to_numpy(), sel.path.tolist()
        feat_dir = out / "feats"
    feat_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(paths)} images, device={dev}")

    for name in args.backbones:
        f_out = feat_dir / f"{name}.npz"
        if f_out.exists():
            print(f"[skip] {name} (exists)"); continue
        t0 = time.time()
        model, tf = build(name)
        model.to(dev)
        dl = DataLoader(ImageList(args.root, paths, tf), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=(dev == "cuda"))
        feats = []
        for x in dl:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                feats.append(model(x.to(dev, non_blocking=True)).float().cpu())
        feats = torch.cat(feats).numpy().astype(np.float16)
        np.savez(f_out, feats=feats, row_id=row_id, **extra)
        print(f"[done] {name}: {feats.shape} in {time.time() - t0:.0f}s")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
