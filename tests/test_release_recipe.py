"""Contract checks against recorded run c6043bd4, independent of training defaults."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from bone_age_training.config import TrainConfig
from bone_age_training.data import build_transforms
from bone_age_training.model import double_mae
from bone_age_training.training import BoneAgeModule, IterationSampler

ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((ROOT / "docs/recipe-provenance.json").read_text())


def test_effective_release_settings_match_recorded_run():
    config = TrainConfig.load(ROOT / "configs/release.json")
    old = RECORD["recorded_recipe"]
    assert config.backbone == old["backbone"]
    assert config.batch_size == old["batch_size"] == 32
    assert config.steps_per_epoch == old["num_iterations_per_epoch"] == 2000
    assert config.epochs == old["num_epochs"] == 10
    assert config.image_size == old["image_height"] == old["image_width"] == 512
    assert config.learning_rate == old["optimizer_params"]["lr"]
    assert config.warmup_fraction == old["scheduler_params"]["pct_start"]
    assert config.div_factor == old["scheduler_params"]["div_factor"]
    assert config.final_div_factor == old["scheduler_params"]["final_div_factor"]
    assert config.epochs * config.steps_per_epoch == old["scheduler_params"]["total_steps"]
    assert config.precision == old["trainer"]["precision"] == "16-mixed"
    assert config.sync_batchnorm == old["trainer"]["sync_batchnorm"]
    assert config.workers == old["num_workers"]
    assert old["num_input_channels"] == 2 and old["num_classes"] == [1, 240]
    assert old["loss_params"] == {"reg_weight": 1.0, "cls_weight": 1.0}


def test_loss_is_equal_weight_mean_absolute_error_in_months():
    logits = torch.full((2, 240), -torch.inf)
    logits[0, 20] = 0
    logits[1, 60] = 0
    loss = double_mae(torch.tensor([4.0, 12.0]), logits, torch.tensor([10.0, 40.0]))
    # Regression MAE=17; classification expected-month MAE=15; equal mean=16.
    assert loss.item() == pytest.approx(16.0)


def actual_attributes(obj, expected):
    out = {"type": obj.__class__.__name__}
    for key in expected:
        if key == "type":
            continue
        if key == "transforms":
            out[key] = [
                actual_attributes(o, e) for o, e in zip(obj.transforms, expected[key], strict=True)
            ]
        else:
            value = getattr(obj, key)
            if key == "max_size" and not isinstance(value, (list, tuple)):
                value = [value]
            # Canonicalize nested tuples to the stored JSON representation.
            out[key] = json.loads(json.dumps(value))
    return out


@pytest.mark.parametrize("train,name", [(True, "train_transforms"), (False, "val_transforms")])
def test_all_recorded_augmentation_parameters_match(train, name):
    expected = RECORD["recorded_augmentations"][name]
    assert actual_attributes(build_transforms(train), expected) == expected


def test_preserved_five_way_split_and_distributed_epoch_counts():
    path = ROOT / "data/splits.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RECORD["splits_sha256"]
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len({r["pid"] for r in rows}) == len(rows) == 14036
    counts = Counter(int(r["fold"]) for r in rows)
    assert set(counts) == {0, 1, 2, 3, 4}
    old = RECORD["recorded_recipe"]
    n_val = counts[old["fold"]] * 2
    n_train = len(rows) * 2 - n_val
    assert n_val == old["n_val"] and n_train == old["n_train"]
    samples = old["num_iterations_per_epoch"] * old["batch_size"]
    ranks = [list(IterationSampler(n_train, samples, rank=i, world_size=2)) for i in range(2)]
    assert len(ranks[0]) == len(ranks[1]) == 64000
    assert sum(map(len, ranks)) * old["num_epochs"] == 1280000


def test_model_heads_pooling_and_optimizer_parameters():
    config = TrainConfig(backbone="bone_age_testnet", pretrained=False)
    module = BoneAgeModule(config.to_dict())
    old = RECORD["recorded_recipe"]
    assert [head.out_features for head in module.model.linear] == old["num_classes"]
    assert module.model.pooling.p.item() == old["pool_params"]["p"]
    assert module.model.dropout.p == old["dropout"]
    optimizer = module.configure_optimizers()["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == old["optimizer_params"]["lr"]
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
