# Geo Satellites: a four-model image classification ensemble

A 20-class satellite-image classification solution for the **[Week 5 course competition](https://www.kaggle.com/competitions/week-5-competition-geo-satellites-classification)** at Innopolis University. The final ensemble combines two ResNet101 runs, InceptionV3 and VGG16-BN, with ImageNet initialization, EMA and rotation averaging at inference.

**Historical public accuracy reported in the original experiment notes: 0.96230.** Those notes specify the same ensemble weights used here. The score has not been independently checked against the final leaderboard or reproduced during repository preparation; final private score and rank are not recorded here.

[Русское описание](README.ru.md) · [Final notebook](solution.ipynb) · [Script](solution.py) · [Result provenance](reports/recorded_results.json)

## Final model

| Component | Backbone | Seed | Image size | Epochs | Rotation views | Ensemble weight |
|---|---|---:|---:|---:|---:|---:|
| R42 | ResNet101 | 42 | 256×256 | 15 | 2 | 0.153 |
| R1337 | ResNet101 | 1337 | 256×256 | 15 | 4 | 0.357 |
| INC | InceptionV3 | 42 | 299×299 | 15 | 4 | 0.340 |
| VGG | VGG16-BN | 42 | 256×256 | 9 | 1 | 0.150 |

ResNet101 starts from torchvision's ImageNet-1K V2 weights; InceptionV3 and VGG16-BN use ImageNet-1K V1. The classification heads are replaced for 20 classes. VGG also uses global average pooling and a compact linear head.

- **Training:** AdamW with separate backbone/head learning rates, warmup and cosine decay, label smoothing, MixUp/CutMix and an exponential moving average of weights with decay 0.999. ResNet and Inception use 12 mixing epochs followed by 3 clean epochs. VGG uses 9 mixing epochs with microbatch 6 and accumulation 3, giving an effective batch of 18.
- **Parallelism:** GPU 0 trains R42 then R1337; GPU 1 trains Inception then VGG. Inference also uses two parallel stages.
- **Prediction:** each model averages class probabilities over its configured rotations; the four probability arrays are blended with the weights above, then `argmax` selects the class. R42 uses 0°/180°; R1337 and Inception use 0°/90°/180°/270°; VGG uses 0°. The final TTA configuration does not use reflections.

## Data and evaluation

The implementation expects the original competition files and validates their sizes:

| Split | Images | Use in this final notebook |
|---|---:|---|
| `train.csv` and `train/` | 4,999 | Final training |
| `val.csv` and `val/` | 699 | Merged with train for final training |
| `test.csv` and `test/` | 8,299 | Submission inference |

CSV image identifiers are in `id`; labeled files additionally contain integer `label` values from 0 to 19. Image filenames must match `id`. The sample submission must preserve the same ID order as `test.csv`.

**This is a final-refit pipeline.** It trains on all 5,698 labeled images after the model configuration has been chosen. It does not re-run model selection or provide a held-out validation score. The printed training `top1` values, especially on mixed batches, must not be presented as validation accuracy.

The historical score above comes from the introduction to `week5_FINAL4_R42_R1337_INC_VGG_GRID.ipynb`, recorded on 2026-10-10. It reports 0.96230 for the 0.153/0.357/0.340/0.150 ensemble. The available final notebooks have no saved execution outputs, and no independent link between a rerun of this notebook and that submission has been established. Details and limitations are retained in [recorded_results.json](reports/recorded_results.json).

## Run on Kaggle

1. Obtain the competition data through Kaggle and attach them to a notebook.
2. Upload `solution.ipynb`, select **GPU T4 ×2**, and enable Internet if torchvision's pretrained weights are not cached.
3. Ensure the CSVs and image directories are together in the data root, with `sample_submission.csv` or `sample_submission_example.csv`.
4. Run all cells. The output is `final_submission.csv` in the working directory.

The original data discovery checks the current directory, `../input`, and their immediate child directories. If data are nested more deeply, change only the input-root discovery or use the optional timed notebook below.

For a **Linux environment with two CUDA GPUs**, the same executable code is also exported as a script:

```bash
python -m pip install -r requirements.txt
python solution.py
```

Run from the repository root with the competition data in the root or a direct child directory. PyTorch must be a CUDA build compatible with the installed driver, and torchvision must match it. The dependency list is intentionally not a historical lock: the original final notebook did not record package versions.

### Timing variant

[notebooks/timed_retrain.ipynb](notebooks/timed_retrain.ipynb) is the later instrumented version of the same four-model setup. It searches Kaggle input directories recursively, can download the competition data with `kagglehub`, and writes timing/event reports after a real run. It requires Kaggle competition access and uses the same two-GPU execution model. Its outputs are cleared.

The original code enforces inference ≤120 seconds and training-plus-inference ≤45 minutes. These are **runtime checks**, not measured guarantees in this repository. The original total excludes initial data and pretrained-weight downloads; the timed variant separately records the full notebook duration.

## Practical limitations and provenance

- Two CUDA GPUs are required by the preserved implementation; a single-GPU machine cannot run it unchanged.
- The full GPU training/inference run was not repeated during repository preparation. Seeds are fixed, but parallel execution and `cudnn.benchmark=True` do not promise bitwise-identical reruns.
- Model states are kept in memory rather than written as checkpoints. Restarting the kernel requires retraining.
- Competition images, prediction CSVs and pretrained/model weights are excluded from this repository.

This is a **CHIMIN PRIME team course project**, archived as part of Vladimir Garanin's portfolio. AI assistance was used in the coursework and repository preparation. The canonical implementation comes from `final_solution.ipynb`; the only executable change is adding the missing `Subset` import for the optional small-test-loader branch. Documentation, section headings, metadata cleanup and the script export were added during packaging. [Implementation provenance](reports/provenance.json) records source hashes and the exact scope of changes.
