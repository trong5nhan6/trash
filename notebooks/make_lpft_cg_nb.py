"""Generates kaggle_lpft_cg.ipynb: LP-FT + CG-SHOT-LP for all 7 backbones (2 Kaggle sessions)."""
import json
from pathlib import Path

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# LP-FT + CG-SHOT-LP — 7 backbone × 3 seed

**Train-time** LP-FT (LoRA cho ViT, stage cuối cho CNN) → **test-time** CG-SHOT-LP trên feature đã fine-tune.
Mỗi run báo: `lp` (= B0), `ft` (LP-FT), `ft_probe`, **`ft_cg_hard`**, **`ft_cg_soft`**, `ft_shot_lprop_auto`, `ft_t3a_ens`, `ft_cg_hard_t3a`.

**Toàn bộ ≈ 17–20 giờ GPU T4 → chia 2 phiên** (mỗi phiên ~6 giờ trên Kaggle 2×T4, dưới giới hạn 12 giờ):
* Phiên 1: `SESSION = 1` → DINOv2-L (3 seed, chia 2 GPU) + SigLIP
* Phiên 2: `SESSION = 2` → DINOv2-B, ResNet-50, EfficientNet-B0, CLIP, ConvNeXt-T

Mỗi phiên tạo một file `results_lpftcg_session<k>.zip`; gửi cả hai để gộp.
*Kaggle*: GPU T4 x2 + Internet (+ **Add Input** dataset nếu có). *Colab*: chạy được nhưng 1 GPU → chậm gấp đôi.
""")

code("""
import os
IS_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if IS_KAGGLE else "/content"      # Kaggle hoặc Colab
SAVE_TO_DRIVE = not IS_KAGGLE                               # Colab: chép file zip kết quả vào Google Drive
DRIVE_URL = "https://drive.google.com/file/d/1KN9lMekyM5TWDqN5lhT2xqKmGydbkRuW/view?usp=sharing"
DRIVE_ZIP = "/content/drive/MyDrive/datasets.zip"   # Colab: đường dẫn file zip dữ liệu trong Drive của bạn

SESSION   = 1            # <-- 1 hoặc 2 (chạy 2 lần, mỗi lần một phiên)
FT_EPOCHS = 5
# Mỗi phần tử = một chuỗi lệnh chạy trên một GPU: [(backbone, [seeds]), ...]
PLANS = {
    1: [[("dinov2_l", [0, 1])],
        [("dinov2_l", [2]), ("siglip_b16", [0, 1, 2])]],
    2: [[("dinov2_b", [0, 1, 2]), ("resnet50", [0, 1, 2]), ("effnet_b0", [0, 1, 2])],
        [("clip_b16", [0, 1, 2]), ("convnext_t", [0, 1, 2])]],
}
BATCH = {"dinov2_l": 32}  # batch nhỏ hơn cho ViT-L (bộ nhớ T4)

REPO_URL, REPO_DIR = "https://github.com/trong5nhan6/trash.git", f"{WORK}/trash"
DATA_DIR, ROOT, SMALL = "/tmp/data", "/tmp/root", "/tmp/small"
OUT = f"{WORK}/outputs"
""")

code("""
import os, glob, shutil, subprocess, time, torch
if SAVE_TO_DRIVE:                                  # Colab: mount Drive ngay từ đầu (cần bấm cho phép 1 lần)
    from google.colab import drive; drive.mount("/content/drive")
N_GPU, N_CPU = torch.cuda.device_count(), os.cpu_count()
print("GPUs:", N_GPU, "| CPUs:", N_CPU)
if os.path.isdir(REPO_DIR): subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"], check=True)
else: subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
!pip install -q "timm>=1.0.9"
!git -C {REPO_DIR} log --oneline -1
plan = PLANS[SESSION] if N_GPU >= 2 else [sum(PLANS[SESSION], [])]   # 1 GPU: chạy tuần tự
BACKBONES = sorted({bb for chain in plan for bb, _ in chain})
print("SESSION", SESSION, "->", plan)
""")

code("""
# Dữ liệu + manifest
# Nguồn dữ liệu theo thứ tự: (1) Kaggle Dataset đã gắn vào notebook ("Add Input"),
# (2) Colab: file zip trong Google Drive đã mount, (3) tải trực tiếp link Drive bằng wget (resume + thử lại).
os.makedirs(DATA_DIR, exist_ok=True)
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
        time.sleep(120)
        for g, log, p in procs:
            try:   # progress display only: must never stop the run
                lines = open(log, errors="ignore").read().strip().splitlines() if os.path.exists(log) else []
                done = [l for l in lines if l.startswith("[done]")]
                last = (done[-1:] or lines[-1:] or [""])[0]
                print(f"  {(time.time()-t0)/60:5.0f} min | chain{g}: {len(done)} runs done | {last[:140]}")
            except Exception as e:
                print(f"  (monitor: {e})")
    for g, log, p in procs:
        if p.returncode != 0:
            print(open(log).read()[-3000:]); raise RuntimeError(f"{stage} chain{g} failed")
    print(f"[{stage}] done in {(time.time()-t0)/60:.1f} min")
W = max(1, N_CPU // len(plan))
""")

code("""
# Feature đóng băng (cho đầu LP) + dinov2_b (để lọc ảnh gần trùng), rồi lọc ảnh gần trùng
need = sorted(set(BACKBONES) | {"dinov2_b"})
split = [need[i::len(plan)] for i in range(len(plan))]
run_on_gpus([[f"extract.py --root {ROOT} --out {OUT} --workers {W} --backbones {' '.join(b)}"] for b in split], "A_extract")
!cd {REPO_DIR}/code && python neardup.py --out {OUT} --feats dinov2_b --thr 0.95
!cd {REPO_DIR}/code && python preresize.py --root {ROOT} --out {OUT} --dst {SMALL}
""")

code("""
# LP-FT + CG-SHOT-LP (GPU). Mỗi run lưu ngay 1 file -> nếu phiên bị ngắt vẫn giữ được phần đã xong.
ft = lambda bb, seeds: (f"finetune.py --root {SMALL} --out {OUT} --backbone {bb} --seeds {' '.join(map(str, seeds))} "
                        f"--epochs {FT_EPOCHS} --bs {BATCH.get(bb, 64)} --workers {W} --cg --tag lpftcg")
run_on_gpus([[ft(bb, s) for bb, s in chain] for chain in plan], f"FT_session{SESSION}")
""")

code("""
# Tổng hợp phiên này (bảng đầy đủ khi gộp 2 phiên)
!cd {REPO_DIR}/code && python summarize.py --out {OUT} --methods lpftcg
import pandas as pd
pd.set_option("display.width", 220)
f = f"{OUT}/results_methods/lpftcg/summary.csv"
if os.path.exists(f):
    s = pd.read_csv(f)
    order = ["lp", "ft", "ft_probe", "ft_cg_hard", "ft_cg_soft", "ft_cg_hard_t3a", "ft_shot_lprop_auto", "ft_t3a_ens"]
    s["o"] = s.variant.map({v: i for i, v in enumerate(order)})
    display(s.sort_values(["backbone", "o"])[["backbone", "variant", "n_seeds", "in_domain_F1", "ood_F1", "gap",
                                              "LODO_avg_F1", "ood_ECE", "d_ood_vs_base", "d_lodo_vs_base"]])
""")

code("""
tag = f"lpftcg_session{SESSION}"
tmp = "/tmp/results_only"; shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(OUT, tmp, ignore=shutil.ignore_patterns("feats*"))
shutil.make_archive(f"{WORK}/results_{tag}", "zip", tmp)
!ls -lh {WORK}/*.zip
if SAVE_TO_DRIVE:
    dst = "/content/drive/MyDrive/taimma_results"; os.makedirs(dst, exist_ok=True)
    for z in glob.glob(f"{WORK}/*.zip"): shutil.copy(z, dst); print("copied", os.path.basename(z), "->", dst)
""")

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).with_name("kaggle_lpft_cg.ipynb")
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", p, len(cells), "cells")
