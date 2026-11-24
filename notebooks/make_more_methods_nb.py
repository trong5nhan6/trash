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

*Settings*: **GPU T4 x2**, **Internet On** → *Run All*. M6 chỉ chạy phần còn thiếu theo `FT_PLAN` (mặc định: dinov2_b seed 2 + resnet50 seed 0–2).
Mọi bước resume được. Notebook độc lập: tự tải data, trích feature gốc, lọc ảnh gần trùng.
""")

code("""
import os
IS_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if IS_KAGGLE else "/content"      # Kaggle hoặc Colab
SAVE_TO_DRIVE = not IS_KAGGLE                               # Colab: chép file zip kết quả vào Google Drive
DRIVE_URL = "https://drive.google.com/file/d/1KN9lMekyM5TWDqN5lhT2xqKmGydbkRuW/view?usp=sharing"
DRIVE_ZIP = "/content/drive/MyDrive/datasets.zip"   # Colab: đường dẫn file zip dữ liệu trong Drive của bạn
SEEDS     = [0, 1, 2]          # M4, M5 (CPU, rẻ)
FT_EPOCHS = 5
# M6: backbone -> seeds cần fine-tune. Lần chạy trước (trash-method-1) đã xong siglip_b16 seed 0-2
# và dinov2_b seed 0-1, nên mặc định chỉ chạy phần còn thiếu. Chạy đủ từ đầu:
# FT_PLAN = {"siglip_b16": [0, 1, 2], "dinov2_b": [0, 1, 2], "resnet50": [0, 1, 2]}
FT_PLAN   = {"dinov2_b": [2], "resnet50": [0, 1, 2]}
RUN       = {"ens": True, "rich": True, "lpft": True}

REPO_URL, REPO_DIR = "https://github.com/trong5nhan6/trash.git", f"{WORK}/trash"
DATA_DIR, ROOT, SMALL = "/tmp/data", "/tmp/root", "/tmp/small"
OUT = f"{WORK}/outputs"
""")

code("""
import os, glob, shutil, subprocess, time, torch
if SAVE_TO_DRIVE:                                  # Colab: mount Drive ngay từ đầu (cần bấm cho phép 1 lần)
    from google.colab import drive; drive.mount("/content/drive")
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
# Nguồn dữ liệu theo thứ tự: (1) Kaggle Dataset đã gắn vào notebook ("Add Input"),
# (2) Colab: file zip trong Google Drive đã mount, (3) tải trực tiếp link Drive bằng wget (resume + thử lại).
has = lambda base: bool(glob.glob(f"{base}/**/TrashBox_train_set", recursive=True))
SEARCH = "/kaggle/input" if IS_KAGGLE and has("/kaggle/input") else DATA_DIR
if SEARCH == DATA_DIR and not has(DATA_DIR):
    z = f"{DATA_DIR}/data.zip"
    if SAVE_TO_DRIVE and os.path.exists(DRIVE_ZIP):
        print("copy from mounted Drive:", DRIVE_ZIP); shutil.copy(DRIVE_ZIP, z)
    else:
        fid = DRIVE_URL.split("/d/")[1].split("/")[0]
        url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
        for attempt in range(5):
            r = subprocess.run(["wget", "-q", "-c", "-O", z, url])
            if r.returncode == 0 and os.path.getsize(z) > 1e9: break
            if os.path.exists(z) and os.path.getsize(z) < 1e8: os.remove(z)   # HTML error page, not data
            print(f"download attempt {attempt + 1} failed, retry in 60 s"); time.sleep(60)
        assert os.path.exists(z) and os.path.getsize(z) > 1e9, (
            "Không tải được dữ liệu từ Drive (Google giới hạn lượt tải). Cách chắc chắn: upload thư mục "
            "datasets thành Kaggle Dataset rồi 'Add Input' vào notebook, hoặc trên Colab đặt DRIVE_ZIP.")
    subprocess.run(["unzip", "-q", "-o", z, "-d", DATA_DIR], check=True); os.remove(z)
print("data source:", SEARCH)
MARKERS = {"TrashNet": "dataset-resized", "RealWaste": "realwaste-main",
           "TrashBox": "TrashBox_train_set", "Garbage12": "garbage_classification"}
os.makedirs(ROOT, exist_ok=True)
for name, marker in MARKERS.items():
    hits = [h for h in glob.glob(f"{SEARCH}/**/{marker}", recursive=True) if os.path.isdir(h)]
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
            try:   # progress display only: must never stop the run
                last = (open(log, errors="ignore").read().strip().splitlines()[-1:] if os.path.exists(log) else []) or [""]
                print(f"  {(time.time()-t0)/60:4.0f} min | chain{g}: {last[0][:150]}")
            except Exception as e:
                print(f"  (monitor: {e})")
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
    ft = lambda bb, seeds: (f"finetune.py --root {SMALL} --out {OUT} --backbone {bb} "
                            f"--seeds {' '.join(map(str, seeds))} --epochs {FT_EPOCHS} --workers {W}")
    chains = [[], []] if N_GPU >= 2 else [[]]          # 1 GPU (Colab): fine-tune tuần tự
    for i, (bb, seeds) in enumerate(FT_PLAN.items()): chains[i % len(chains)].append(ft(bb, seeds))
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
shutil.make_archive(f"{WORK}/results_{tag}", "zip", tmp)
shutil.make_archive(f"{WORK}/full_{tag}", "zip", OUT)
!ls -lh {WORK}/*.zip
if SAVE_TO_DRIVE:
    dst = "/content/drive/MyDrive/taimma_results"; os.makedirs(dst, exist_ok=True)
    for z in glob.glob(f"{WORK}/*.zip"): shutil.copy(z, dst); print("copied", os.path.basename(z), "->", dst)
""")

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).with_name("kaggle_more_methods_2xT4.ipynb")
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", p, len(cells), "cells")
