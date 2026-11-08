import csv
import json

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from bone_age_training.cli import main
from bone_age_training.inference import crop_manifest, cropped_image, evaluate


class CropModel:
    def preprocess(self, x):
        return x

    def __call__(self, x, shape):
        assert shape.tolist() == [[6, 8]]
        return torch.tensor([[-2.0, 1.0, 6.0, 3.0]])

    def eval(self):
        return self

    def to(self, device):
        return self


def test_crop_clips_coords_and_checks_empty_boxes():
    image = np.arange(48, dtype=np.uint8).reshape(6, 8)
    np.testing.assert_array_equal(cropped_image(image, CropModel(), "cpu"), image[1:4, :4])

    class Empty(CropModel):
        def __call__(self, x, shape):
            return torch.tensor([[10.0, 10.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="empty"):
        cropped_image(image, Empty(), "cpu")


def test_cli_prepare_matches_fold_ids_and_crop_pipeline(tmp_path, monkeypatch):
    labels = tmp_path / "labels.csv"
    splits = tmp_path / "splits.csv"
    manifest = tmp_path / "manifest.csv"
    (tmp_path / "train").mkdir()
    pd.DataFrame({"pid": ["001"], "female": [1], "bone_age": [120], "subset": ["train"]}).to_csv(
        labels, index=False
    )
    pd.DataFrame({"pid": ["001"], "fold": [0], "subset": ["train"]}).to_csv(splits, index=False)
    main(["prepare", "--labels", str(labels), "--splits", str(splits), "--output", str(manifest)])
    cv2.imwrite(str(tmp_path / "train/001.png"), np.arange(48, dtype=np.uint8).reshape(6, 8))
    monkeypatch.setattr(
        "bone_age_training.inference.AutoModel.from_pretrained", lambda *a, **kw: CropModel()
    )
    crop_manifest(manifest, tmp_path, "cpu")
    assert cv2.imread(str(tmp_path / "cropped_train_plus_valid/001.png"), 0).shape == (3, 4)
    main(["check-data", "--manifest", str(manifest), "--data-root", str(tmp_path)])


def test_evaluation_predicts_months_and_keeps_input_table(tmp_path, monkeypatch):
    class AgeModel(CropModel):
        def __call__(self, x, sex):
            return torch.tensor([120.0])

    labels = tmp_path / "test.csv"
    pd.DataFrame({"pid": ["001"], "female": [1], "bone_age": [100]}).to_csv(labels, index=False)
    before = labels.read_bytes()
    (tmp_path / "test").mkdir()
    cv2.imwrite(str(tmp_path / "test/001.png"), np.zeros((6, 8), dtype=np.uint8))
    monkeypatch.setattr(
        "bone_age_training.inference.AutoModel.from_pretrained", lambda *a, **kw: AgeModel()
    )
    output = tmp_path / "predictions.csv"
    evaluate("local-model", labels, tmp_path, "test", None, False, "cpu", output)
    with output.open() as f:
        result = list(csv.DictReader(f))
    assert float(result[0]["absolute_error"]) == 20
    assert labels.read_bytes() == before


def test_ddp_worker_uses_existing_rank_zero_output(tmp_path, monkeypatch):
    from test_training_export import fixture_data

    from bone_age_training import training

    fixture_data(tmp_path)
    output = tmp_path / "fold0"
    output.mkdir()
    sentinel = output / "effective_config.json"
    sentinel.write_text('{"rank":0}')

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(training.L, "Trainer", FakeTrainer)
    config = training.TrainConfig(backbone="bone_age_testnet", pretrained=False, workers=0)
    training.train(
        config, tmp_path / "manifest.csv", tmp_path, tmp_path / "ref.png", 0, 1, "cpu", output
    )
    assert json.loads(sentinel.read_text()) == {"rank": 0}


def test_reference_download_is_pinned_and_integrity_checked(tmp_path, monkeypatch):
    from bone_age_training import assets

    source = tmp_path / "download.png"
    cv2.imwrite(str(source), np.zeros((4, 4), dtype=np.uint8))
    calls = []

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(source)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(assets, "REFERENCE_SHA256", assets.sha256(source))
    output = tmp_path / "reference.png"
    main(["reference", "--output", str(output)])
    assert output.read_bytes() == source.read_bytes()
    assert calls[0][2]["revision"] == assets.RELEASE_REVISION
    monkeypatch.setattr(assets, "REFERENCE_SHA256", "wrong")
    with pytest.raises(ValueError, match="does not match"):
        main(["reference", "--output", str(tmp_path / "bad.png")])
    assert not (tmp_path / "bad.png").exists()


def test_resume_rejects_changed_data_before_restoration(tmp_path):
    from test_training_export import fixture_data

    from bone_age_training.assets import data_identity
    from bone_age_training.training import TrainConfig, train

    fixture_data(tmp_path)
    config = TrainConfig(backbone="bone_age_testnet", pretrained=False, workers=0)
    checkpoint = tmp_path / "resume.ckpt"
    torch.save(
        {
            "training_fold": 0,
            "hyper_parameters": {"config": config.to_dict()},
            "training_data": data_identity(tmp_path / "manifest.csv", tmp_path / "ref.png"),
        },
        checkpoint,
    )
    cv2.imwrite(str(tmp_path / "ref.png"), np.ones((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="same training manifest and histogram reference"):
        train(
            config,
            tmp_path / "manifest.csv",
            tmp_path,
            tmp_path / "ref.png",
            0,
            1,
            "cpu",
            tmp_path / "run",
            resume=checkpoint,
        )
    assert not (tmp_path / "run").exists()
