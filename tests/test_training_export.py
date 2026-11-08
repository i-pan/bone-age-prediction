import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModel

from bone_age_training.config import TrainConfig
from bone_age_training.data import prepare_manifest
from bone_age_training.export import classification_state, export
from bone_age_training.model import BoneAgeNet, double_mae, expected_age
from bone_age_training.schedule import LinearWarmupCosineAnnealingLR
from bone_age_training.training import IterationSampler, train


def test_expected_age_and_loss_backpropagate_to_both_heads():
    config = TrainConfig(backbone="bone_age_testnet", pretrained=False)
    model = BoneAgeNet(config)
    reg, logits = model(torch.rand(2, 2, 16, 16) * 255)
    loss = double_mae(reg, logits, torch.tensor([20.0, 100.0]))
    loss.backward()
    assert model.linear[0].weight.grad.abs().sum() > 0
    assert model.linear[1].weight.grad.abs().sum() > 0
    uniform = torch.zeros(3, 240)
    torch.testing.assert_close(expected_age(uniform), torch.full((3,), 119.5))


def test_release_scheduler_boundaries():
    p = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([p], lr=0.0003)
    schedule = LinearWarmupCosineAnnealingLR(
        optimizer,
        max_lr=0.0003,
        total_steps=20000,
        pct_start=0.1,
        div_factor=100,
        final_div_factor=1000,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)
    schedule.last_epoch = 2000
    assert schedule.get_lr()[0] == pytest.approx(3e-4)
    schedule.last_epoch = 20000
    assert schedule.get_lr()[0] == pytest.approx(3e-9)


def test_distributed_sampling_exact_iterations_and_epoch_shuffle():
    rank0 = IterationSampler(10, 4, rank=0, world_size=2)
    rank1 = IterationSampler(10, 4, rank=1, world_size=2)
    first = list(rank0)
    assert len(first) == len(list(rank1)) == 4
    assert not set(first) & set(rank1)
    rank0.set_epoch(1)
    assert list(rank0) != first
    repeated = list(IterationSampler(3, 20))
    assert len(repeated) == 20 and set(repeated) == {0, 1, 2}


def fixture_data(root):
    for d in ["train", "cropped_train_plus_valid"]:
        (root / d).mkdir()
    labels = pd.DataFrame(
        {
            "pid": ["a", "b", "c"],
            "bone_age": [20, 80, 140],
            "female": [0, 1, 0],
            "imgfile": ["train/a.png", "train/b.png", "train/c.png"],
        }
    )
    labels.to_csv(root / "labels.csv", index=False)
    pd.DataFrame({"pid": ["a", "b", "c"], "fold": [0, 1, 2], "subset": ["train"] * 3}).to_csv(
        root / "splits.csv", index=False
    )
    prepare_manifest(root / "labels.csv", root / "splits.csv", root / "manifest.csv")
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for pid in ["a", "b", "c"]:
        for d in ["train", "cropped_train_plus_valid"]:
            cv2.imwrite(str(root / d / f"{pid}.png"), image)
    cv2.imwrite(str(root / "ref.png"), image)


def test_train_resume_export_and_automodel_roundtrip(tmp_path):
    fixture_data(tmp_path)
    config = TrainConfig(
        backbone="bone_age_testnet",
        pretrained=False,
        image_size=512,
        steps_per_epoch=1,
        epochs=1,
        batch_size=2,
        workers=0,
    )
    output = tmp_path / "fold0"
    train(
        config,
        tmp_path / "manifest.csv",
        tmp_path,
        tmp_path / "ref.png",
        0,
        1,
        "cpu",
        output,
        limit_val_batches=1,
    )
    checkpoint = torch.load(output / "last.ckpt", weights_only=True)
    assert checkpoint["training_fold"] == 0
    assert checkpoint["global_step"] == 1
    assert checkpoint["optimizer_states"]
    model = BoneAgeNet(config).eval()
    state = {
        k.removeprefix("model."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state)
    # Actual CPU training plus three independent fold-labelled payloads validates export routing.
    paths = [output / "last.ckpt"]
    for fold in [1, 2]:
        payload = dict(checkpoint, training_fold=fold)
        weights = dict(checkpoint["state_dict"])
        weights["model.linear.1.bias"] = (
            weights["model.linear.1.bias"] + torch.arange(240) * fold / 100
        )
        payload["state_dict"] = weights
        path = tmp_path / f"fold{fold}.ckpt"
        torch.save(payload, path)
        paths.append(path)
    export(paths, tmp_path / "ref.png", tmp_path / "export")
    loaded = AutoModel.from_pretrained(tmp_path / "export", trust_remote_code=True).eval()
    image = torch.rand(2, 1, 12, 12) * 255
    female = torch.tensor([0.0, 1.0])
    inp = torch.cat([image, female[:, None, None, None].expand_as(image) * 255], dim=1)
    with torch.inference_mode():
        _, logits = model(inp)
        torch.testing.assert_close(loaded.net0(image, female, return_logits=True), logits)
        members = [getattr(loaded, f"net{i}")(image, female) for i in range(3)]
        torch.testing.assert_close(loaded(image, female), torch.stack(members).mean(0))
    provenance = json.loads((tmp_path / "export/training_provenance.json").read_text())
    assert [r["fold"] for r in provenance] == [0, 1, 2]
    assert not any(Path(r["checkpoint"]).is_absolute() for r in provenance)
    # Resume with the identical schedule: no additional steps should run.
    train(
        config,
        tmp_path / "manifest.csv",
        tmp_path,
        tmp_path / "ref.png",
        0,
        1,
        "cpu",
        output,
        limit_val_batches=1,
        resume=output / "last.ckpt",
    )


def test_export_rejects_repeated_fold(tmp_path):
    config = TrainConfig(backbone="bone_age_testnet", pretrained=False)
    model = BoneAgeNet(config)
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.ckpt"
        torch.save(
            {
                "state_dict": {"model." + k: v for k, v in model.state_dict().items()},
                "hyper_parameters": {"config": config.to_dict()},
                "training_fold": 0,
            },
            p,
        )
        paths.append(p)
    cv2.imwrite(str(tmp_path / "ref.png"), np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="fold metadata"):
        export(paths, tmp_path / "ref.png", tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_original_convnext_architecture_forward():
    model = BoneAgeNet(TrainConfig(pretrained=False)).eval()
    with torch.inference_mode():
        regression, logits = model(torch.zeros(1, 2, 64, 64))
    assert regression.shape == (1,) and logits.shape == (1, 240)
    assert classification_state(
        {"state_dict": {"model." + k: v for k, v in model.state_dict().items()}}
    )["linear.weight"].shape == (240, 768)


def test_unfinished_resume_restores_remaining_updates_and_lr(tmp_path, monkeypatch):
    import lightning as L

    from bone_age_training import training

    fixture_data(tmp_path)
    config = TrainConfig(
        backbone="bone_age_testnet",
        pretrained=False,
        image_size=16,
        steps_per_epoch=2,
        epochs=2,
        batch_size=2,
        workers=0,
    )
    actual_trainer = L.Trainer

    class StopAfterEpochZero(L.Callback):
        def on_train_epoch_end(self, trainer, module):
            if trainer.current_epoch == 0:
                trainer.should_stop = True

    def stop_early(**kwargs):
        kwargs["callbacks"] = [StopAfterEpochZero(), *kwargs["callbacks"]]
        kwargs["enable_progress_bar"] = False
        kwargs["enable_model_summary"] = False
        return actual_trainer(**kwargs)

    monkeypatch.setattr(training.L, "Trainer", stop_early)
    output = tmp_path / "interrupted"
    args = (config, tmp_path / "manifest.csv", tmp_path, tmp_path / "ref.png", 0, 1, "cpu")
    train(*args, output, limit_val_batches=1)
    saved = torch.load(output / "last.ckpt", weights_only=True)
    assert saved["global_step"] == 2 and saved["epoch"] == 0
    assert saved["lr_schedulers"][0]["last_epoch"] == 2
    monkeypatch.setattr(training.L, "Trainer", actual_trainer)
    train(*args, output, limit_val_batches=1, resume=output / "last.ckpt")
    resumed = torch.load(output / "last.ckpt", weights_only=True)
    assert resumed["global_step"] == 4 and resumed["epoch"] == 1
    assert resumed["lr_schedulers"][0]["last_epoch"] == 4
    assert resumed["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(3e-9)
    for state in resumed["optimizer_states"][0]["state"].values():
        assert state["step"].item() == 4


def test_resume_initialization_skips_pretrained_download(monkeypatch):
    from bone_age_training import training

    observed = []
    actual = training.BoneAgeNet

    def record(config):
        observed.append(config.pretrained)
        return actual(config)

    monkeypatch.setattr(training, "BoneAgeNet", record)
    config = TrainConfig(backbone="bone_age_testnet", pretrained=True)
    module = training.BoneAgeModule(config.to_dict(), initialize_pretrained=False)
    assert observed == [False]
    assert module.hparams["config"]["pretrained"] is True


def test_export_rejects_mixed_training_data_and_wrong_reference(tmp_path):
    from bone_age_training.assets import sha256

    config = TrainConfig(backbone="bone_age_testnet", pretrained=False)
    model = BoneAgeNet(config)
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), np.zeros((4, 4), dtype=np.uint8))
    identity = {"manifest_sha256": "same-manifest", "reference_sha256": sha256(ref)}
    paths = []
    for fold in range(3):
        path = tmp_path / f"fold{fold}.ckpt"
        torch.save(
            {
                "state_dict": {"model." + k: v for k, v in model.state_dict().items()},
                "hyper_parameters": {"config": config.to_dict()},
                "training_fold": fold,
                "training_data": dict(identity),
            },
            path,
        )
        paths.append(path)
    cv2.imwrite(str(ref), np.ones((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="Export reference differs"):
        export(paths, ref, tmp_path / "wrong-reference")
    payload = torch.load(paths[2], weights_only=True)
    payload["training_data"]["manifest_sha256"] = "different-manifest"
    torch.save(payload, paths[2])
    with pytest.raises(ValueError, match="share their training manifest"):
        export(paths, ref, tmp_path / "mixed-data")
