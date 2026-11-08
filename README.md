# Bone age prediction

Training companion for [ianpan/bone-age](https://huggingface.co/ianpan/bone-age): a three-model ConvNeXtV2-Tiny ensemble for pediatric bone-age estimation from hand radiographs and patient sex. Predictions are in months.

The training model has regression and 240-class heads. Both heads optimize mean absolute error, with the classification prediction computed as the expected month under its softmax distribution. Export retains the classification head used by the published model.

## Install

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra cu124         # CUDA 12.4 PyTorch wheels
# or: uv sync --extra cpu     # CPU development and tests
uv run --no-sync bone-age --help
```

Use `--no-sync` in the commands below after selecting your environment. `uv.lock` pins the dependencies.

## Data

Obtain the RSNA Pediatric Bone Age Challenge training, validation, and test data through the dataset provider. Keep radiographs outside this repository. The training set uses 14,036 training/validation images; the challenge test set has 200 images.

Arrange your data root as:

```text
bone-age/
  train/<pid>.png
  valid/<pid>.png
  test/<pid>.png
  labels.csv
  test.csv
```

`labels.csv` contains `pid,bone_age,female,subset`, with bone age in months, `female` encoded as 0 or 1, and `subset` as `train` or `valid`. `test.csv` contains `pid,bone_age,female`. IDs and subset names must agree with the original split assignments in `data/splits.csv`. The CLI also accepts an explicit relative `imgfile` column in the training labels.

Prepare the paired uncropped/cropped manifest, download the published histogram reference, and generate crops with [ianpan/bone-age-crop](https://huggingface.co/ianpan/bone-age-crop):

```bash
uv run --no-sync bone-age prepare --labels /path/to/bone-age/labels.csv \
  --output /path/to/bone-age/manifest.csv
uv run --no-sync bone-age reference --output /path/to/bone-age/ref_img.png
uv run --no-sync bone-age crop --manifest /path/to/bone-age/manifest.csv \
  --data-root /path/to/bone-age --device cuda
uv run --no-sync bone-age check-data --manifest /path/to/bone-age/manifest.csv \
  --data-root /path/to/bone-age
```

Each image has an uncropped and a cropped row with the same fold assignment. See [data/README.md](data/README.md) for split details. Crop and reference commands download only when explicitly invoked.

The crop model and reference image use fixed Hugging Face revisions recorded in `src/bone_age_training/assets.py`. The reference download also verifies the published image's SHA-256 hash.

## Train

The release recipe in `configs/release.json` uses:

| Setting | Value |
| --- | --- |
| Backbone | ImageNet-pretrained `convnextv2_tiny` |
| Input | 512 × 512 grayscale plus a patient-sex channel |
| Pooling / dropout | Learnable GeM (p=3) / 0.1 |
| Heads | Regression (1) and month classification (240) |
| Loss | Equal-weight mean of regression MAE and classification expected-age MAE |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.01 |
| Schedule | 20,000 optimizer steps; 10% linear warmup then cosine decay |
| Epochs | 10 × 2,000 steps |
| Batch | 32 per GPU; 64 total on two GPUs |
| Precision | FP16 mixed precision |
| Ensemble | Independently train folds 0, 1, and 2 of the preserved five-fold split |

Cropping and histogram matching are each applied to approximately half the training examples. Augmentation also includes flips, shifts, scaling, rotation, blur, noise, brightness, and contrast. Validation applies histogram matching to both image variants.

```bash
uv run --no-sync bone-age train --config configs/release.json \
  --manifest /path/to/bone-age/manifest.csv --data-root /path/to/bone-age \
  --reference /path/to/bone-age/ref_img.png --fold 0 --devices 2 --output outputs/fold0
```

Repeat with `--fold 1 --output outputs/fold1` and `--fold 2 --output outputs/fold2`. Each fold trains on all images outside its held-out partition, including partitions 3 and 4. Validation reports regression and classification MAE in months. Keep `last.ckpt` for ensemble export; `best.ckpt` tracks regression validation MAE. CSV logs, effective configuration, and checkpoint state are saved under each fold output.

To continue an interrupted run, repeat its training command with `--resume outputs/fold0/last.ckpt`. Resume restores the optimizer and schedule, skips pretrained-weight downloads, and checks that the fold, effective configuration, manifest, and histogram reference match the checkpoint. Checkpoints record SHA-256 identities for the manifest and reference; export checks those identities across all three members.

For a CPU smoke run, use `--devices 1 --accelerator cpu --steps-per-epoch 1 --epochs 1 --batch-size 2 --image-size 64 --workers 0 --no-pretrained --limit-val-batches 1`. This command performs a real short optimization run only when you invoke it.

## Export and evaluate

Export the three classification heads as a local Hugging Face model directory:

```bash
uv run --no-sync bone-age export --checkpoints \
  outputs/fold0/last.ckpt outputs/fold1/last.ckpt outputs/fold2/last.ckpt \
  --reference /path/to/bone-age/ref_img.png --output exports/bone-age
uv run --no-sync bone-age evaluate --model exports/bone-age \
  --labels /path/to/bone-age/test.csv --data-root /path/to/bone-age \
  --image-dir test --reference /path/to/bone-age/ref_img.png \
  --crop --device cuda --output outputs/test-predictions.csv
```

Export produces Safetensors weights, Transformers configuration/modeling code, the histogram reference, and provenance for the input checkpoints. It does not upload files. Evaluation reports mean absolute error in months and saves one prediction per test image. Omit `--crop` or `--reference` to evaluate the other preprocessing combinations.

The exported directory supports `AutoModel.from_pretrained("exports/bone-age", trust_remote_code=True)`. Use `model.preprocess(grayscale_image)` before converting to an NCHW float tensor. Pass a binary `female` tensor separately. Individual members are accessible as `model.net0`, `model.net1`, and `model.net2`.

## Development

```bash
uv run --no-sync pytest
uv run --no-sync ruff check .
```

Adapted from [Ian Pan's SKP](https://github.com/i-pan/SKP). Training recipe: `boneage.cfg_female_channel_reg_cls_match_hist_cropped_uncropped`, run `c6043bd4`. Published models: [bone-age](https://huggingface.co/ianpan/bone-age) and [bone-age-crop](https://huggingface.co/ianpan/bone-age-crop). Code is MIT-licensed; dataset acquisition is separate.
