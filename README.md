# Cross-dataset waste classification (TAIMMA 2027)

Baseline **B0**: frozen backbone features + linear probe, across 4 waste datasets,
unified 5-class label space (cardboard, glass, metal, paper, plastic).

## Data layout (not in this repo)

```
<DATA_ROOT>/
  TrashNet/dataset-resized/<class>/*.jpg
  RealWaste/realwaste-main/RealWaste/<Class>/*.jpg
  TrashBox/TrashBox_train_set/<class>/*.jpg
  TrashBox/TrashBox-testandvalid/TrashBox_testandvalid_set/{val,test}/<class>/*.jpg
  Garbage12/garbage_classification/<class>/*.jpg
```

## Pipeline

| Step | Script | GPU | Output |
|---|---|---|---|
| 1. Scan, verify, exact dedup (md5) | `code/prepare.py` | no | `outputs/manifest.csv`, `table1_counts.csv`, `cross_dataset_duplicates.csv` |
| 2. Frozen features, 7 backbones | `code/extract.py` | yes | `outputs/feats/<backbone>.npz` |
| 3. Linear probe, single-source 4x4 + LODO | `code/probe.py --seed k` | no | `outputs/results/seed<k>/` |

```bash
pip install -r requirements.txt
python code/prepare.py --root $DATA_ROOT --out outputs
python code/extract.py --root $DATA_ROOT --out outputs            # resumable, skips finished backbones
python code/probe.py   --out outputs --seed 0
```

### Kaggle
Upload the data folder as a Kaggle Dataset, enable GPU (T4), then in a notebook:
```bash
!git clone https://github.com/trong5nhan6/trash.git && cd trash && pip -q install -r requirements.txt
!cd trash && python code/prepare.py --root /kaggle/input/<your-dataset-slug> --out /kaggle/working/outputs
!cd trash && python code/extract.py --root /kaggle/input/<your-dataset-slug> --out /kaggle/working/outputs
!cd trash && python code/probe.py --out /kaggle/working/outputs --seed 0
```

## Protocol notes
- **Dedup:** exact duplicates across datasets are kept only in the earliest-published dataset
  (TrashNet 2016 > Garbage12 2021 > TrashBox 2022 > RealWaste 2023). Within a dataset, duplicates with
  conflicting labels are dropped entirely.
  Found: 2,352 TrashNet images inside Garbage12 and 1,005 inside TrashBox.
- **Split:** stratified 70/10/20 per dataset x class; the seed controls the split. TrashBox's official
  split is pooled and re-split like the others.
- **Model selection:** probe C chosen by macro-F1 on the *source* validation split only.
- **Metrics:** accuracy, macro-F1, ECE (15 bins).
