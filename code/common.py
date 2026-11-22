"""Shared config: dataset locations, unified label space, deterministic splits."""
from pathlib import Path
import numpy as np

# Unified 5-class label space shared by all four datasets.
CORE = ["cardboard", "glass", "metal", "paper", "plastic"]
CORE_TO_ID = {c: i for i, c in enumerate(CORE)}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# dirs: folders (relative to --root) whose immediate subfolders are class folders.
# map : original class folder name -> unified label. Unlisted classes -> "other"
#       (kept in the manifest for later open-set experiments, excluded from the 5-class task).
DATASETS = {
    "TrashNet": {
        "domain": "lab",
        "dirs": ["TrashNet/dataset-resized"],
        "map": {c: c for c in CORE},                       # 'trash' -> other
    },
    "RealWaste": {
        "domain": "real",
        "dirs": ["RealWaste/realwaste-main/RealWaste"],
        "map": {"Cardboard": "cardboard", "Glass": "glass", "Metal": "metal",
                "Paper": "paper", "Plastic": "plastic"},   # Food Organics, Misc, Textile, Vegetation -> other
    },
    "TrashBox": {
        "domain": "web",
        # Official train/val/test are pooled and re-split like the other datasets.
        "dirs": ["TrashBox/TrashBox_train_set",
                 "TrashBox/TrashBox-testandvalid/TrashBox_testandvalid_set/val",
                 "TrashBox/TrashBox-testandvalid/TrashBox_testandvalid_set/test"],
        "map": {c: c for c in CORE},                       # e-waste, medical -> other
        # NOTE: TrashBox 'paper' contains tetra pak and paper cups (multi-material).
    },
    "Garbage12": {
        "domain": "web",
        "dirs": ["Garbage12/garbage_classification"],
        "map": {"cardboard": "cardboard", "metal": "metal", "paper": "paper", "plastic": "plastic",
                "brown-glass": "glass", "green-glass": "glass", "white-glass": "glass"},
    },
}
DATASET_NAMES = list(DATASETS)

# Publication order, used to resolve exact duplicates shared by several datasets:
# the copy in the earliest (original) dataset is kept, later copies are dropped.
# TrashNet 2016, Garbage12 (Kaggle) 2021, TrashBox 2022, RealWaste 2023.
DEDUP_PRIORITY = ["TrashNet", "Garbage12", "TrashBox", "RealWaste"]


def make_split(datasets, labels, seed, frac=(0.7, 0.1, 0.2)):
    """Stratified (per dataset x label) train/val/test split. Returns array of 'train'/'val'/'test'."""
    datasets, labels = np.asarray(datasets), np.asarray(labels)
    split = np.empty(len(labels), dtype=object)
    rng = np.random.default_rng(seed)
    for d in np.unique(datasets):
        for c in np.unique(labels[datasets == d]):
            idx = np.where((datasets == d) & (labels == c))[0]
            idx = idx[rng.permutation(len(idx))]
            n_tr = int(round(frac[0] * len(idx)))
            n_va = int(round(frac[1] * len(idx)))
            split[idx[:n_tr]] = "train"
            split[idx[n_tr:n_tr + n_va]] = "val"
            split[idx[n_tr + n_va:]] = "test"
    return split


def resolve_root(root):
    root = Path(root)
    missing = [d for cfg in DATASETS.values() for d in cfg["dirs"] if not (root / d).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing folders under {root}:\n  " + "\n  ".join(missing))
    return root
