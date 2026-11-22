"""Method M2 prerequisite: image embeddings in the joint image-text space + text class weights.

Uses open_clip (projected, L2-normalised embeddings) so zero-shot logits are valid; the timm
features used by B0 are pre-projection and cannot be compared with text.
Output: outputs/emb/<name>.npz with feats [N,D] (normalised), row_id, logit_scale,
        text_<set> [5,D] for every prompt set in prompts.py.

Usage: python extract_clip.py --root <data root> --out outputs [--models clip_b16 siglip_b16]
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch, open_clip
from torch.utils.data import DataLoader
from extract import ImageList
from prompts import prompt_sets

MODELS = {"clip_b16": ("ViT-B-16", "openai"), "siglip_b16": ("ViT-B-16-SigLIP", "webli")}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out); (out / "emb").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(out / "manifest.csv")
    sel = man[(man.keep == 1) & (man.core == 1)]
    row_id, paths = sel.index.to_numpy(), sel.path.tolist()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for name in args.models:
        f_out = out / "emb" / f"{name}.npz"
        if f_out.exists():
            print(f"[skip] {name}"); continue
        t0 = time.time()
        arch, tag = MODELS[name]
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
        tok = open_clip.get_tokenizer(arch)
        model = model.to(dev).eval()
        text = {}
        for set_name, per_class in prompt_sets().items():
            w = []
            for prompts in per_class:
                e = torch.nn.functional.normalize(model.encode_text(tok(prompts).to(dev)).float(), dim=-1)
                w.append(torch.nn.functional.normalize(e.mean(0), dim=-1))
            text[f"text_{set_name}"] = torch.stack(w).cpu().numpy()
        dl = DataLoader(ImageList(args.root, paths, preprocess), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=(dev == "cuda"))
        feats = []
        for x in dl:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                f = model.encode_image(x.to(dev, non_blocking=True)).float()
            feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        np.savez(f_out, feats=torch.cat(feats).numpy().astype(np.float16), row_id=row_id,
                 logit_scale=float(model.logit_scale.exp().item()), **text)
        print(f"[done] {name}: {len(paths)} images in {time.time() - t0:.0f}s")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
