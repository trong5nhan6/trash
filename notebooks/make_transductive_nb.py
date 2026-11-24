"""Generates kaggle_transductive_methods.ipynb: data-analysis-driven methods (single backbone, CPU after
feature extraction). Positive: label propagation, SHOT centroids, SHOT+LP. Negative controls: label-shift EM,
sub-class probe, LEACE nuisance erasure, background-propensity reweighting, T3A reference."""
import json
from pathlib import Path

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Data-driven, single-backbone methods (transductive, label-free on target)

| Method | Problem measured in the data | Fix |
|---|---|---|
| **LP** label propagation (`labelprop`) | same-class target images cluster together but the source boundary is misplaced; wrong predictions are low-confidence | propagate source posteriors over the target kNN graph; (k, α) and temperature chosen on **source val** |
| **SHOT** centroid refinement (`cluster`) | class centroids move between datasets (same-class cross-dataset distance ≈ between-class distance) | re-estimate class centroids on the unlabelled target, nearest-centroid, ensemble with source probe |
| **SHOT→LP** (`labelprop`, variant `shot_lprop_auto`) | both | seed the graph with SHOT-refined posteriors |
| controls | prior shift · multi-modal classes · background shortcut | EM label-shift (`prior`), sub-class probe (`subclass`), LEACE erasure (`erase`), propensity reweighting (`reweight`), T3A (`tta`) |

*Settings*: GPU T4 x2 (chỉ để trích feature, ~10 phút), Internet On → *Run All*. Tổng ~1–1.5 giờ.
""")

code("""
import os
IS_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if IS_KAGGLE else "/content"      # Kaggle hoặc Colab
SAVE_TO_DRIVE = not IS_KAGGLE                               # Colab: chép file zip kết quả vào Google Drive
DRIVE_URL = "https://drive.google.com/file/d/1KN9lMekyM5TWDqN5lhT2xqKmGydbkRuW/view?usp=sharing"
DRIVE_ZIP = "/content/drive/MyDrive/datasets.zip"   # Colab: đường dẫn file zip dữ liệu trong Drive của bạn
SEEDS     = [0, 1, 2]
METHODS   = ["labelprop", "cluster", "tta", "prior", "subclass", "erase", "reweight"]
JOBS      = 3            # tiến trình CPU song song (giữ thấp để không hết RAM)

REPO_URL, REPO_DIR = "https://github.com/trong5nhan6/trash.git", f"{WORK}/trash"
DATA_DIR, ROOT, OUT = "/tmp/data", "/tmp/root", f"{WORK}/outputs"
""")

code("""
import os, glob, shutil, subprocess, time, torch
if SAVE_TO_DRIVE:                                  # Colab: mount Drive ngay từ đầu (cần bấm cho phép 1 lần)
    from google.colab import drive; drive.mount("/content/drive")
N_GPU, N_CPU = torch.cuda.device_count(), os.cpu_count()
print("GPUs:", N_GPU, "| CPUs:", N_CPU)
if os.path.isdir(REPO_DIR): subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"], check=True)
else: subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
!pip install -q gdown "timm>=1.0.9"
!git -C {REPO_DIR} log --oneline -1
""")

code("""
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
    link = f"{ROOT}/{name}"
    if os.path.islink(link): os.remove(link)
    os.symlink(os.path.dirname(sorted(hits, key=len)[0]), link)
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(f"{OUT}/manifest.csv"):
    !cd {REPO_DIR}/code && python prepare.py --root {ROOT} --out {OUT}
""")

code("""
# Trích feature gốc (2 GPU song song) -> lọc ảnh gần trùng -> đặc trưng nuisance (CPU)
W = max(1, N_CPU // 2)
procs = []
for g, bbs in enumerate([["siglip_b16", "resnet50"], ["dinov2_b"]]):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % max(N_GPU, 1)))
    procs.append(subprocess.Popen(["python", "extract.py", "--root", ROOT, "--out", OUT, "--workers", str(W),
                                   "--backbones", *bbs], cwd=f"{REPO_DIR}/code", env=env))
assert all(p.wait() == 0 for p in procs), "extraction failed"
!cd {REPO_DIR}/code && python neardup.py --out {OUT} --feats dinov2_b --thr 0.95
!cd {REPO_DIR}/code && python nuisance.py --root {ROOT} --out {OUT}
""")

code("""
S = " ".join(map(str, SEEDS))
for m in METHODS:
    print(f"\\n################ {m} ################", flush=True)
    !cd {REPO_DIR}/code && python run_methods.py --out {OUT} --method {m} --seeds {S} --jobs {JOBS}
""")

code("""
!cd {REPO_DIR}/code && python summarize.py --out {OUT} --methods {" ".join(METHODS)}
""")

code("""
tmp = "/tmp/results_only"; shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(OUT, tmp, ignore=shutil.ignore_patterns("feats*"))
shutil.make_archive(f"{WORK}/results_transductive", "zip", tmp)
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
p = Path(__file__).with_name("kaggle_transductive_methods.ipynb")
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", p, len(cells), "cells")
