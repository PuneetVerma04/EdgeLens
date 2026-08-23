# Defect Detection — EdgeLens

Serving repo: [EdgeLens](https://github.com/PuneetVerma04/EdgeLens) (this repo trains the model; EdgeLens serves it)

Deep learning pipeline for industrial defect classification using ResNet50 and MobileNetV2.
Supports two datasets: casting product images (binary) and PCB surface defects (9-class).

## Known issues — read before citing any accuracy number here

- **`TrainingModelTensorFlow.ipynb` still validates on its training set.** `train_ds` and `test_ds`
  (cells 11–12) are both built from the same `data_dir`, with no `validation_split`/`subset`. Every
  `val_accuracy` figure in that notebook's committed output — up to 96.31% — is measured on data the
  model trained on and carries no information about generalization.
- **`DefectResNet.ipynb` (PyTorch) still has an unfixed data-leakage bug of its own.**
  `val_dataset.dataset.transform = val_transforms` mutates the *shared* underlying `ImageFolder`, so
  the augmentation pipeline (`RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`) is silently
  disabled for training too — the reported 99.62% validation accuracy does not reflect the augmented
  training the notebook describes. `random_split` is also called with no seed, so the split (and the
  number) is not reproducible. There is no held-out test set — only train/val — so 99.62% is a
  validation figure, not a test figure.
- **The `src/` package below is a corrected rewrite of the PyTorch pipeline — but neither notebook
  uses it yet.** `src/dataset.py` fixes both bugs above (seeded split, val transforms applied via a
  separate `ImageFolder` instance instead of mutating the shared one) and `src/model.py` uses the
  non-deprecated `weights=` API instead of `pretrained=True`. It is committed and ready to import,
  but `DefectResNet.ipynb` has not been rewired to call it — the notebook still runs its own,
  original, buggy inline code path.

Bottom line: **no accuracy figure in this repo currently reflects real generalization.** Rewiring
`DefectResNet.ipynb` to use `src/` and giving the TF notebook a real split are open work — tracked in
the project backlog, not done here (this pass is documentation-only).

## Datasets

| Dataset | Classes | Images | Format |
|---------|---------|--------|--------|
| [Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | 2 (def_front, ok_front) | 1300 | ImageFolder |
| [DsPCBSD](https://www.kaggle.com/datasets/akhatova/pcb-defects) | 9 (SH, SP, SC, OP, MB, HB, CS, CFO, BMFO) | 10 259 | COCO + YOLO |

## Setup

```bash
pip install -r requirements.txt
```

Update dataset paths in `config.yaml` before running anything.

## Project Structure

```
├── config.yaml                     # All paths and hyperparameters
├── src/
│   ├── dataset.py                  # DataLoader factories for both datasets
│   ├── model.py                    # ResNet50 and MobileNetV2 model factories
│   ├── trainer.py                  # Training loop, EarlyStopping
│   └── utils.py                    # Plotting, evaluation, config loading
├── scripts/
│   └── prepare_pcb_crops.py        # Extract COCO bounding boxes → ImageFolder crops
├── DefectResNet.ipynb              # PyTorch ResNet50 training notebook
└── TrainingModelTensorFlow.ipynb   # TensorFlow MobileNetV2 training notebook
```

## PCB Dataset — Data Preparation

The PCB dataset ships as COCO object-detection annotations. Before training the classifier,
extract per-defect crops from bounding boxes:

```bash
python scripts/prepare_pcb_crops.py
# or with custom paths:
python scripts/prepare_pcb_crops.py --raw-path D:/Datasets/DsPCBSD --crops-path D:/Datasets/DsPCBSD/crops
```

This creates `crops/train/<class>/` and `crops/val/<class>/` directories that the PyTorch
DataLoader can consume directly via `torchvision.datasets.ImageFolder`.

## Training

### PyTorch (ResNet50)

`src/` is a corrected, config-driven rewrite of the training pipeline (see Known Issues above) —
intended usage once a notebook is wired to it:

```python
from src.utils import load_config
from src.dataset import load_casting_loaders
from src.model import get_resnet50

cfg = load_config("config.yaml")
train_loader, val_loader, class_names = load_casting_loaders(
    cfg["datasets"]["casting"]["path"], cfg["datasets"]["casting"]["input_size"],
    cfg["training"]["batch_size"], cfg["training"]["val_split"], cfg["training"]["num_workers"],
)
```

**`DefectResNet.ipynb` does not currently do this.** It still contains its own original, inline
data-loading and training code — the bugs described in Known Issues above are live in the notebook
as committed.

### TensorFlow (MobileNetV2)

Open `TrainingModelTensorFlow.ipynb`. Update `dataset_path` at the top to point to
the appropriate dataset folder.

## Saved Models

| File | Description |
|------|-------------|
| `outputs/models/best_<run>.pth` | Best checkpoint by val loss (PyTorch) |
| `outputs/models/final_<run>.pth` | End-of-training weights (PyTorch) |
| `defect_detection_mobilenetv2.h5` | Best checkpoint (TensorFlow/Keras) |
| `final_defect_detection_resnet.pth` | Legacy model from initial training run |

## Other notes

- `num_workers` is set to `0` in `config.yaml` — required on Windows to avoid DataLoader
  multiprocessing errors. Increase on Linux for faster data loading.
- MobileNetV2 in the TF notebook uses `input_shape=(300, 300, 3)` but ImageNet weights are
  loaded for `(224, 224)` — a torchvision warning is expected and harmless.

See "Known issues" at the top of this file for the accuracy/leakage caveats — those are the ones
that matter before citing any number from this repo.

## License

MIT — see [LICENSE](LICENSE). Academic/portfolio project; not validated for production use.
