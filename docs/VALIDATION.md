# Validation

Validated locally on 2026-11-08 using Python 3.12 and the locked CPU environment.

- 53 automated tests passed.
- Synthetic CPU training/checkpoint tests passed, including an unfinished two-epoch run that resumed with the correct remaining optimizer updates and terminal learning rate.
- Three fold-labelled checkpoint payloads exported into a portable Transformers/Safetensors directory. Reloading through `AutoModel` preserved classification logits and ensemble predictions.
- The ConvNeXtV2-Tiny training model completed an inference forward pass with 1 regression and 240 classification outputs.
- Data preparation reproduced all 28,072 rows of the preserved training manifest, matching image IDs, paths, labels, sex, fold, and source subset for all 14,036 source images.
- All referenced training/cropped image files were present during the data check.
- The histogram reference downloaded from `ianpan/bone-age` matched the training reference byte for byte.
- Split validation covers patient leakage, paired original/cropped rows, missing images, label/subset agreement, and path containment.
- The scheduler was checked at its initial, peak, and terminal learning rates.
- A two-process CPU/Gloo validation probe checked global metric reduction over unequal batches and checkpoint participation by both ranks.
- Recipe regression tests compare model/head settings, loss weights, sampling counts, and all recorded training/validation augmentation parameters with the preserved run configuration.
- Preprocessing assets are revision-pinned; tests check reference integrity and rejection of changed manifest/reference identities on resume and export.
- A fresh read-only reviewer challenged the final consumer and failure paths and reported no remaining material findings within the reviewed scope.
- Ruff lint/format checks and source/wheel builds passed.

Run the automated checks with:

```bash
uv sync --locked --extra cpu
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv build
```

GitHub Actions runs the CPU tests and lint checks for pushes and pull requests.
