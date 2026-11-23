"""Generates kaggle_more_methods_2xT4.ipynb (M4 ensemble, M5 rich features, M6 LP-FT + WiSE)."""
import json
from pathlib import Path

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Methods that should raise in-domain *and* reduce the gap — 2× T4

| | Method | Idea | Device |
|---|---|---|---|
| M4 | Multi-backbone fusion | concat / late fusion of DINOv2-B, SigLIP-B/16, ResNet-50 features | CPU |
| M5 | Rich features | DINOv2-paper probe (CLS of last 4 blocks + mean patch), multi-stage pooling for ResNet, flip averaging | GPU → CPU |
| M6 | LP-FT + WiSE | LP-initialised fine-tuning (LoRA for ViT, layer4 for ResNet) + weight interpolation α = 0.5 | GPU |

*Settings*: **GPU T4 x2**, **Internet On** → *Run All*. Ước tính 3–4 giờ với 3 seed cho M6 (giảm `FT_SEEDS` để nhanh hơn).
Mọi bước resume được. Notebook độc lập: tự tải data, trích feature gốc, lọc ảnh gần trùng.
""")

code("""
DRIVE_URL = "https://drive.google.com/file/d/1KN9lMekyM5TWDqN5lhT2xqKmGydbkRuW/view?usp=sharing"
SEEDS     = [0, 1, 2]          # M4, M5 (CPU, rẻ)
FT_SEEDS  = [0, 1, 2]          # M6 fine-tune (GPU, ~45 phút / seed với 2 GPU)
FT_EPOCHS = 5
FT_BACKBONES = ["siglip_b16", "dinov2_b", "resnet50"]   # chạy lại riêng một backbone: ["resnet50"]
RUN       = {"ens": True, "rich": True, "lpft": True}

REPO_URL, REPO_DIR = "https://github.com/trong5nhan6/trash.git", "/kaggle/working/trash"
DATA_DIR, ROOT, SMALL = "/tmp/data", "/tmp/root", "/tmp/small"
OUT = "/kaggle/working/outputs"
""")

code("""
import os, glob, shutil, subprocess, time, torch
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
N_GPU, N_CPU = torch.cuda.device_count(), os.cpu_count()
print("GPUs:", N_GPU, "| CPUs:", N_CPU)
if os.path.isdir(REPO_DIR): subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"], check=True)
else: subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
!pip install -q gdown "timm>=1.0.9"
!git -C {REPO_DIR} log --oneline -1
""")

code("""
# Dữ liệu + manifest
os.makedirs(DATA_DIR, exist_ok=True)
if not glob.glob(f"{DATA_DIR}/**/TrashBox_train_set", recursive=True):
    z = f"{DATA_DIR}/data.zip"
    subprocess.run(["gdown", "--fuzzy", DRIVE_URL, "-O", z], check=True)
    subprocess.run(["unzip", "-q", "-o", z, "-d", DATA_DIR], check=True); os.remove(z)
MARKERS = {"TrashNet": "dataset-resized", "RealWaste": "realwaste-main",
           "TrashBox": "TrashBox_train_set", "Garbage12": "garbage_classification"}
os.makedirs(ROOT, exist_ok=True)
for name, marker in MARKERS.items():
    hits = [h for h in glob.glob(f"{DATA_DIR}/**/{marker}", recursive=True) if os.path.isdir(h)]
    assert hits, f"missing {marker}"
    link = f"{ROOT}/{name}"
    if os.path.islink(link): os.remove(link)
    os.symlink(os.path.dirname(sorted(hits, key=len)[0]), link)
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(f"{OUT}/manifest.csv"):
    !cd {REPO_DIR}/code && python prepare.py --root {ROOT} --out {OUT}
""")

code("""
def run_on_gpus(chains, stage):
    procs = []
    for g, cmds in enumerate(chains):
        if not cmds: continue
        gpu = g % max(N_GPU, 1)
        log = f"{OUT}/logs/{stage}_gpu{g}.log"; os.makedirs(os.path.dirname(log), exist_ok=True)
        script = " && ".join(f"python {c}" for c in cmds)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        procs.append((g, log, subprocess.Popen(["bash", "-c", f"cd {REPO_DIR}/code && {script} >> {log} 2>&1"], env=env)))
        print(f"[{stage}] GPU{gpu}: {cmds}")
    t0 = time.time()
    while any(p.poll() is None for *_, p in procs):
        time.sleep(60)
        for g, log, p in procs:
            last = (open(log, errors="ignore").read().strip().splitlines()[-1:] if os.path.exists(log) else []) or [""]
            print(f"  {(time.time()-t0)/60:4.0f} min | chain{g}: {last[0][:150]}")
    for g, log, p in procs:
        if p.returncode != 0:
            print(open(log).read()[-3000:]); raise RuntimeError(f"{stage} chain{g} failed")
    print(f"[{stage}] done in {(time.time()-t0)/60:.1f} min")
W = max(1, N_CPU // 2)
""")

code("""
# A. feature gốc (3 backbone) + lọc ảnh gần trùng
run_on_gpus([
    [f"extract.py --root {ROOT} --out {OUT} --workers {W} --backbones siglip_b16 resnet50"],
    [f"extract.py --root {ROOT} --out {OUT} --workers {W} --backbones dinov2_b"],
], "A_extract")
!cd {REPO_DIR}/code && python neardup.py --out {OUT} --feats dinov2_b --thr 0.95
""")

code("""
# B. M5 rich features (GPU)
if RUN["rich"]:
    run_on_gpus([
        [f"extract_rich.py --root {ROOT} --out {OUT} --workers {W} --backbones siglip_b16 resnet50"],
        [f"extract_rich.py --root {ROOT} --out {OUT} --workers {W} --backbones dinov2_b"],
    ], "B_rich")
""")

code("""
# C. M6 LP-FT + WiSE (GPU). Ảnh được thu nhỏ trước (cạnh ngắn 256) để không nghẽn CPU.
if RUN["lpft"]:
    !cd {REPO_DIR}/code && python preresize.py --root {ROOT} --out {OUT} --dst {SMALL}
    S = " ".join(map(str, FT_SEEDS))
    ft = lambda bb: f"finetune.py --root {SMALL} --out {OUT} --backbone {bb} --seeds {S} --epochs {FT_EPOCHS} --workers {W}"
    chains = [[], []]
    for i, bb in enumerate(FT_BACKBONES): chains[i % 2].append(ft(bb))
    run_on_gpus(chains, "C_lpft")
""")

code("""
# D. M4 + M5 đánh giá (CPU)
S = " ".join(map(str, SEEDS))
for method in ["rich", "ens"]:
    if RUN[method]:
        print(f"\\n################ {method} ################")
        !cd {REPO_DIR}/code && python run_methods.py --out {OUT} --method {method} --seeds {S} --jobs {N_CPU}
""")

code("""
# E. Tổng hợp tất cả phương pháp có trong outputs/ -> leaderboard
!cd {REPO_DIR}/code && python summarize.py --out {OUT}
""")

code("""
tag = "more_methods"
tmp = "/tmp/results_only"; shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(OUT, tmp, ignore=shutil.ignore_patterns("feats*", "emb", "fg"))
shutil.make_archive(f"/kaggle/working/results_{tag}", "zip", tmp)
shutil.make_archive(f"/kaggle/working/full_{tag}", "zip", OUT)
!ls -lh /kaggle/working/*.zip
""")

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).with_name("kaggle_more_methods_2xT4.ipynb")
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", p, len(cells), "cells")
