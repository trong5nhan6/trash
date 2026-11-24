"""Generates kaggle_methods_2xT4.ipynb.  python notebooks/make_methods_nb.py"""
import json
from pathlib import Path

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Gap-reduction methods (M1 CLAD-Aug · M2 text anchors · M3 test-time adaptation) — 2× T4

*Settings*: **GPU T4 x2**, **Internet On** → *Run All*. Ước tính 1.5–2.5 giờ.

| Giai đoạn | Việc | Thiết bị |
|---|---|---|
| A | Trích feature gốc (resnet50, dinov2_b, siglip_b16) → lọc ảnh gần trùng (cos ≥ 0.95) | 2 GPU |
| B | Embedding CLIP/SigLIP + text (M2) · tách nền rembg (M1) | 2 GPU |
| C | Tạo ảnh ghép CLAD (`c1`, `full`) | CPU |
| D | Trích feature ảnh ghép | 2 GPU |
| E | Đánh giá M3, M2, M1 với 4×4 + LODO, mọi seed | CPU |

Mỗi bước đều resume được (bỏ qua phần đã xong).
""")

code("""
import os
IS_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if IS_KAGGLE else "/content"      # Kaggle hoặc Colab
SAVE_TO_DRIVE = not IS_KAGGLE                               # Colab: chép file zip kết quả vào Google Drive
DRIVE_URL = "https://drive.google.com/file/d/1KN9lMekyM5TWDqN5lhT2xqKmGydbkRuW/view?usp=sharing"
DRIVE_ZIP = "/content/drive/MyDrive/datasets.zip"   # Colab: đường dẫn file zip dữ liệu trong Drive của bạn
SEEDS     = [0, 1, 2]
RUN       = {"tta": True, "text": True, "clad": True}   # tắt bớt nếu muốn chạy nhanh
CLAD_K    = 3                                           # số ảnh ghép / ảnh gốc

REPO_URL, REPO_DIR = "https://github.com/trong5nhan6/trash.git", f"{WORK}/trash"
DATA_DIR, ROOT     = "/tmp/data", "/tmp/root"
OUT, CLAD_DIR      = f"{WORK}/outputs", "/tmp/clad"
BACKBONES          = ["resnet50", "dinov2_b", "siglip_b16"]
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
# rembg without deps to avoid touching Kaggle's numpy/opencv; onnxruntime-gpu for CUDA
!pip install -q gdown "timm>=1.0.9" open_clip_torch onnxruntime-gpu pooch
!pip install -q --no-deps rembg pymatting
!git -C {REPO_DIR} log --oneline -1
""")

code("""
# Dữ liệu: tải zip từ Drive + chuẩn hoá tên bằng symlink
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
    !cd {REPO_DIR} && python code/prepare.py --root {ROOT} --out {OUT}
""")

code("""
# Helper: chạy một chuỗi lệnh trên mỗi GPU song song, theo dõi log
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
""")

code("""
# A. feature gốc cho 3 backbone, rồi lọc ảnh gần trùng (dùng dinov2_b)
W = max(1, N_CPU // 2)
run_on_gpus([
    [f"extract.py --root {ROOT} --out {OUT} --workers {W} --backbones siglip_b16 resnet50"],
    [f"extract.py --root {ROOT} --out {OUT} --workers {W} --backbones dinov2_b"],
], "A_extract")
!cd {REPO_DIR}/code && python neardup.py --out {OUT} --feats dinov2_b --thr 0.95
""")

code("""
# B. M2 embeddings (open_clip) + M1 tách nền (rembg) — song song 2 GPU
chains = [[], []]
if RUN["text"]: chains[0].append(f"extract_clip.py --root {ROOT} --out {OUT} --workers {W}")
if RUN["clad"]:
    chains[0].append(f"segment.py --root {ROOT} --out {OUT} --shard 0 --nshards 2")
    chains[1].append(f"segment.py --root {ROOT} --out {OUT} --shard 1 --nshards 2")
run_on_gpus(chains, "B_emb_seg")
!head -3 {OUT}/logs/B_emb_seg_gpu1.log 2>/dev/null
""")

code("""
# C. tạo ảnh ghép CLAD (CPU) + xem thử
if RUN["clad"]:
    subprocess.run(["python", "composite.py", "--out", OUT, "--img-out", CLAD_DIR,
                    "--variants", "c1", "full", "--k", str(CLAD_K)], cwd=f"{REPO_DIR}/code", check=True)
    for v in ["c1", "full"]:
        assert os.path.exists(f"{CLAD_DIR}/list_{v}.csv"), f"composite step produced no list_{v}.csv"
    from PIL import Image
    for v in ["c1", "full"]:
        fs = sorted(glob.glob(f"{CLAD_DIR}/{v}/*.jpg"))[::max(1, len(glob.glob(f'{CLAD_DIR}/{v}/*.jpg'))//12)][:12]
        g = Image.new("RGB", (6*170, 2*170), "white")
        for i, f in enumerate(fs): g.paste(Image.open(f).resize((165, 165)), ((i % 6)*170, (i//6)*170))
        print(v); display(g)
""")

code("""
# D. feature của ảnh ghép
if RUN["clad"]:
    mk = lambda bbs, v: f"extract.py --root {CLAD_DIR} --out {OUT} --workers {W} --list {CLAD_DIR}/list_{v}.csv --tag clad_{v} --backbones {bbs}"
    run_on_gpus([[mk("siglip_b16 resnet50", "c1"), mk("siglip_b16 resnet50", "full")],
                 [mk("dinov2_b", "c1"), mk("dinov2_b", "full")]], "D_clad_feats")
""")

code("""
# E. đánh giá các phương pháp (CPU) — mọi seed
S = " ".join(map(str, SEEDS))
for method in ["tta", "text", "clad"]:
    if RUN[method]:
        print(f"\\n################ {method} ################")
        !cd {REPO_DIR}/code && python run_methods.py --out {OUT} --method {method} --seeds {S} --jobs {N_CPU}
""")

code("""
# Bảng so sánh tổng hợp: OOD macro-F1 và chênh lệch so với baseline của chính phương pháp đó
import pandas as pd
pd.set_option("display.width", 220)
cols = ["backbone", "variant", "in_domain_F1", "ood_F1", "gap", "TrashNet->RealWaste_F1",
        "LODO_avg_F1", "ood_ECE", "d_ood_vs_base", "d_lodo_vs_base"]
for method in ["tta", "text", "clad"]:
    f = f"{OUT}/results_methods/{method}/summary.csv"
    if os.path.exists(f):
        print(f"=== {method} ==="); display(pd.read_csv(f)[cols])
""")

code("""
# Nén sau khi chạy xong: results (nhỏ) + full (kèm feature, không kèm ảnh ghép)
tag = "methods_seeds" + "-".join(map(str, SEEDS))
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
p = Path(__file__).with_name("kaggle_methods_2xT4.ipynb")
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", p, len(cells), "cells")
