"""Generates kaggle_gated_methods.ipynb: CG-SHOT-LP (confidence-gated cluster propagation) on any backbone.
Runs on Kaggle (2x T4) or Colab (1 GPU)."""
import json
from pathlib import Path

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# CG-SHOT-LP — Confidence-Gated Cluster Propagation (backbone-agnostic)

Không train lại backbone, không dùng nhãn của bộ đích, chỉ cần feature → áp dụng cho **mọi backbone**.

| Bước | Làm gì | Vấn đề đã đo trong dữ liệu |
|---|---|---|
| 1. Linear probe nguồn (B0) | xác suất `p_b0` và độ tự tin | — |
| 2. SHOT | ước lượng lại tâm lớp trên bộ đích | tâm lớp dịch chỗ giữa các bộ (cùng lớp khác bộ xa ≈ khác lớp cùng bộ) |
| 3. Label propagation | lan truyền xác suất trên đồ thị kNN của bộ đích | ảnh cùng lớp ở bộ đích vẫn tụ thành cụm |
| 4. **Cổng tự tin** | giữ `p_b0` khi tự tin ≥ τ, chỉ sửa phần không chắc | in-domain: ~85% dự đoán tự tin, đúng 97–99%; lỗi OOD tập trung ở dự đoán không chắc |

τ, (k, α) và nhiệt độ đều chọn trên **tập val của bộ nguồn** (τ = ngưỡng nhỏ nhất để dự đoán được giữ đạt ≥ 98% đúng).

Biến thể báo cáo: `b0`, `shot_lprop_auto` (không cổng), `t3a_ens`, **`cg_hard`**, **`cg_soft`**, `cg_hard_t3a`.

*Kaggle*: GPU T4 x2 + Internet. *Colab*: Runtime GPU. → **Run all**. GPU chỉ dùng để trích feature (~15–25 phút); phần còn lại chạy CPU.
""")

code("""
import os
IS_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if IS_KAGGLE else "/content"      # Kaggle hoặc Colab
SAVE_TO_DRIVE = not IS_KAGGLE                               # Colab: chép file zip kết quả vào Google Drive
DRIVE_URL = "https://drive.google.com/file/d/1KN9lMekyM5TWDqN5lhT2xqKmGydbkRuW/view?usp=sharing"
DRIVE_ZIP = "/content/drive/MyDrive/datasets.zip"   # Colab: đường dẫn file zip dữ liệu trong Drive của bạn
SEEDS     = [0, 1, 2]
# mọi backbone đã dùng trong baseline -> chứng minh phương pháp không phụ thuộc backbone
BACKBONES = ["resnet50", "effnet_b0", "convnext_t", "dinov2_b", "dinov2_l", "clip_b16", "siglip_b16"]

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
    subprocess.run(["python", "prepare.py", "--root", ROOT, "--out", OUT], cwd=f"{REPO_DIR}/code", check=True)
""")

code("""
# Trích feature (GPU). 2 GPU -> 2 tiến trình song song; 1 GPU -> tuần tự. Backbone đã có thì bỏ qua.
W = max(1, N_CPU // max(N_GPU, 1))
n_chain = 2 if N_GPU >= 2 else 1
chains = [BACKBONES[i::n_chain] for i in range(n_chain)]
os.makedirs(f"{OUT}/logs", exist_ok=True)
t0, procs = time.time(), []
for g, bbs in enumerate(chains):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
    log = open(f"{OUT}/logs/extract_gpu{g}.log", "a")
    procs.append((g, bbs, subprocess.Popen(["python", "extract.py", "--root", ROOT, "--out", OUT, "--workers", str(W),
                                            "--backbones", *bbs], cwd=f"{REPO_DIR}/code", env=env,
                                           stdout=log, stderr=subprocess.STDOUT)))
    print(f"GPU{g}: {bbs}")
for g, bbs, p in procs:
    p.wait()
    print(f"GPU{g} exit={p.returncode} ({(time.time()-t0)/60:.1f} min)")
    if p.returncode != 0:
        print(open(f"{OUT}/logs/extract_gpu{g}.log").read()[-3000:]); raise RuntimeError("extraction failed")
!ls -lh {OUT}/feats
""")

code("""
# Lọc ảnh gần trùng (cosine >= 0.95 trên feature DINOv2-B) -> cùng giao thức với các thí nghiệm trước
!cd {REPO_DIR}/code && python neardup.py --out {OUT} --feats dinov2_b --thr 0.95
""")

code("""
# CG-SHOT-LP cho mọi backbone x mọi seed (CPU)
S, B = " ".join(map(str, SEEDS)), " ".join(BACKBONES)
!cd {REPO_DIR}/code && python run_methods.py --out {OUT} --method gated --seeds {S} --backbones {B} --jobs {N_CPU}
""")

code("""
# Bảng gọn theo backbone: B0 vs các biến thể (macro-F1 %, mean over seeds)
import pandas as pd
pd.set_option("display.width", 220)
s = pd.read_csv(f"{OUT}/results_methods/gated/summary.csv")
cols = ["backbone", "variant", "in_domain_F1", "ood_F1", "gap", "TrashNet->RealWaste_F1", "LODO_avg_F1",
        "ood_ECE", "d_ood_vs_base", "d_lodo_vs_base"]
order = ["b0", "cg_hard", "cg_soft", "cg_hard_t3a", "shot_lprop_auto", "t3a_ens"]
s["o"] = s.variant.map({v: i for i, v in enumerate(order)})
display(s.sort_values(["backbone", "o"])[cols])
# trung bình trên mọi backbone: thay đổi so với B0
ref = s[s.variant == "b0"].set_index("backbone")
s["d_id"] = s.apply(lambda r: r.in_domain_F1_mean - ref.loc[r.backbone].in_domain_F1_mean, axis=1)
agg = s[s.variant != "b0"].groupby("variant")[["d_id", "d_ood_vs_base", "d_lodo_vs_base"]].mean().round(2)
print("Mean change vs B0 over", s.backbone.nunique(), "backbones (points):"); display(agg.loc[[v for v in order if v in agg.index]])
r = pd.concat(pd.read_csv(f) for f in glob.glob(f"{OUT}/results_methods/gated/seed*/probe_results.csv")).dropna(subset=["tau"])
print("Fraction of predictions kept by the gate (in_domain=1 vs OOD=0):")
display(r.groupby(["backbone", "in_domain"]).kept_frac.mean().unstack().round(3))
""")

code("""
# Nén kết quả (không kèm feature) + bản đầy đủ (kèm feature)
tmp = "/tmp/results_only"; shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(OUT, tmp, ignore=shutil.ignore_patterns("feats*"))
shutil.make_archive(f"{WORK}/results_gated", "zip", tmp)
shutil.make_archive(f"{WORK}/full_gated", "zip", OUT)
!ls -lh {WORK}/*.zip
if SAVE_TO_DRIVE:
    dst = "/content/drive/MyDrive/taimma_results"; os.makedirs(dst, exist_ok=True)
    for z in glob.glob(f"{WORK}/*.zip"): shutil.copy(z, dst); print("copied", os.path.basename(z), "->", dst)
""")

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).with_name("kaggle_gated_methods.ipynb")
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", p, len(cells), "cells")
