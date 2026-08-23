# Legacy — casting classification (archived, not current work)

> **These notebooks are project history. Do not read any accuracy figure here as a result.**
> The current EdgeLens pipeline is NEU-DET surface-defect **detection**, in [`training/`](../)
> one level up. Nothing in this folder feeds the served model.

This is the contents of the former `DefectDetectionEdgeLens` repository, moved here so the
project tells one end-to-end story in one place. It is the earlier phase of the work: binary
casting-product classification (OK / Defective) with ResNet50 and MobileNetV2, superseded by
the 6-class YOLO detection pipeline.

Imported from `DefectDetectionEdgeLens` at commit `08a872f`.

## Why the numbers here are unreliable

Both notebooks have defects that invalidate their reported metrics. These are real bugs found
by audit, not hypotheticals, and neither has been fixed in the notebooks:

**`TrainingModelTensorFlow.ipynb` — train/validation leak.** `train_ds` (cell 11) and `test_ds`
(cell 12) are both built from the same `data_dir` with no `validation_split` / `subset` argument,
so they are the same images; `test_ds` is the entire dataset, not a held-out portion. Cell 17
then passes `validation_data=test_ds`. Every `val_accuracy` in the committed output — up to
**96.31%** — is measured on data the model trained on and says nothing about generalization.

**`DefectResNet.ipynb` — silently disabled augmentation.** In cell 5,
`val_dataset.dataset.transform = val_transforms` mutates the *shared* `ImageFolder` that
`random_split` handed to both subsets, so it overwrites the training transform too. The
augmentation the notebook appears to apply (`RandomHorizontalFlip`, `RandomRotation`,
`ColorJitter`) never runs. The reported **99.62%** therefore does not describe the pipeline the
code claims to implement. `random_split` is also called with no seed, so the split — and the
number — is not reproducible, and there is no held-out test set at all (train/val only).

**Bottom line: no accuracy figure in this folder reflects real generalization.** They are
retained for provenance, so the project's history is visible rather than quietly deleted.

## What is worth looking at

`src/` is a corrected rewrite of the PyTorch pipeline: `src/dataset.py` fixes both bugs above
(seeded split; val transforms applied through a separate `ImageFolder` instance instead of
mutating the shared one), and `src/model.py` uses the non-deprecated `weights=` API. It was
committed and ready to import, but **`DefectResNet.ipynb` was never rewired to call it** — the
notebook still runs its own original, buggy inline code path. That rewiring was never completed,
and will not be: the project moved to detection instead.

`README.original.md` is the repository's own README as it stood at import, kept unedited.

## Contents

| Path | What it is |
|---|---|
| `DefectResNet.ipynb` | PyTorch ResNet50 casting classifier — has the augmentation bug above |
| `TrainingModelTensorFlow.ipynb` | TF/Keras MobileNetV2 casting classifier — has the leak above |
| `src/` | Corrected PyTorch pipeline; never wired into the notebooks |
| `scripts/prepare_pcb_crops.py` | Crop extraction for the PCB dataset (9-class, exploratory) |
| `config.yaml` | Dataset paths and training hyperparameters |
| `requirements.txt` | Dependencies for the legacy pipeline only |
| `README.original.md` | The original repository README, unedited |
