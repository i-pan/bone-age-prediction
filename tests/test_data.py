from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from bone_age_training.data import BoneAgeDataset, check_manifest, prepare_manifest


@pytest.fixture
def sample(tmp_path):
    labels = tmp_path / "labels.csv"
    splits = tmp_path / "splits.csv"
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "pid": ["001", "002"],
            "bone_age": [0, 240],
            "female": [1, 0],
            "imgfile": ["train/001.png", "valid/002.png"],
        }
    ).to_csv(labels, index=False)
    pd.DataFrame({"pid": ["002", "001"], "fold": [1, 0], "subset": ["valid", "train"]}).to_csv(
        splits, index=False
    )
    prepare_manifest(labels, splits, manifest)
    for folder in ("train", "valid", "cropped_train_plus_valid"):
        (tmp_path / folder).mkdir()
    for relative in pd.read_csv(manifest).imgfile:
        cv2.imwrite(str(tmp_path / relative), np.arange(32, dtype=np.uint8).reshape(4, 8))
    reference = tmp_path / "reference.png"
    cv2.imwrite(str(reference), np.full((4, 8), 180, dtype=np.uint8))
    return tmp_path, labels, splits, manifest, reference


def test_prepare_preserves_patient_splits_and_views(sample):
    root, _, _, manifest, _ = sample
    df = pd.read_csv(manifest, dtype={"pid": str})
    assert len(df) == 4
    assert df.groupby("pid").size().to_dict() == {"001": 2, "002": 2}
    assert df.groupby("pid").fold.first().to_dict() == {"001": 0, "002": 1}
    assert df.groupby("pid").subset.first().to_dict() == {"001": "train", "002": "valid"}
    assert check_manifest(manifest, root) == {"rows": 4, "patients": 2, "folds": {0: 1, 1: 1}}


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_exact_split_coverage(sample, change):
    root, labels, splits, _, _ = sample
    df = pd.read_csv(splits, dtype={"pid": str})
    if change == "missing":
        df = df.iloc[:1]
    else:
        extra = df.iloc[[0]].copy()
        if change == "extra":
            extra["pid"] = "999"
        df = pd.concat([df, extra])
    df.to_csv(splits, index=False)
    with pytest.raises(ValueError, match="coverage|unique"):
        prepare_manifest(labels, splits, root / "bad.csv")


@pytest.mark.parametrize(
    "column,value", [("fold", 2), ("subset", "other"), ("female", 0), ("bone_age", 12)]
)
def test_repeated_patient_metadata_and_leakage(sample, column, value):
    root, _, _, manifest, reference = sample
    df = pd.read_csv(manifest, dtype={"pid": str})
    df.loc[2, column] = value
    df.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="inconsistent"):
        check_manifest(manifest, root)
    with pytest.raises(ValueError, match="inconsistent"):
        BoneAgeDataset(manifest, root, reference, 0, True)


@pytest.mark.parametrize(
    "bad_path", ["/tmp/escape.png", "../escape.png", "train/../../x", "C:\\x.png", "train\\x.png"]
)
def test_unsafe_paths(sample, bad_path):
    root, _, _, manifest, _ = sample
    df = pd.read_csv(manifest)
    df.loc[0, "imgfile"] = bad_path
    df.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="path"):
        check_manifest(manifest, root)


def test_symlink_escape(sample, tmp_path):
    root, _, _, manifest, _ = sample
    image = root / "train/001.png"
    image.unlink()
    image.symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="escapes"):
        check_manifest(manifest, root)


def test_duplicate_paths(sample):
    root, _, _, manifest, _ = sample
    df = pd.read_csv(manifest)
    df.loc[2, "imgfile"] = df.loc[0, "imgfile"]
    df.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        check_manifest(manifest, root)


def test_missing_images_and_reference(sample):
    root, _, _, manifest, reference = sample
    dataset = BoneAgeDataset(manifest, root, reference, 0, False, 8)
    (root / "train/001.png").unlink()
    with pytest.raises(FileNotFoundError):
        check_manifest(manifest, root)
    with pytest.raises(FileNotFoundError):
        dataset[0]
    with pytest.raises(FileNotFoundError):
        BoneAgeDataset(manifest, root, root / "missing.png", 0, False)


def test_crop_requirement_can_be_disabled(sample):
    root, _, _, manifest, _ = sample
    df = pd.read_csv(manifest).iloc[:2]
    df.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="cropped"):
        check_manifest(manifest, root)
    assert check_manifest(manifest, root, require_cropped=False)["rows"] == 2


def test_validation_matches_histogram_and_centers_padding(sample):
    root, _, _, manifest, reference = sample
    item = BoneAgeDataset(manifest, root, reference, 0, False, 16)[0]
    image = item["image"][0]
    # A 4x8 image becomes 8x16, with four zero rows above and below.
    assert torch.all(image[4:12] == 180)
    assert torch.all(image[:4] == 0)
    assert torch.all(image[12:] == 0)
    assert torch.all(item["image"][1] == 255)


def test_duplicate_label_ids_rejected(sample):
    root, labels, splits, _, _ = sample
    df = pd.read_csv(labels, dtype={"pid": str})
    pd.concat([df, df.iloc[[0]]]).to_csv(labels, index=False)
    with pytest.raises(ValueError, match="unique"):
        prepare_manifest(labels, splits, root / "bad.csv")


@pytest.mark.parametrize("train", [True, False])
def test_fold_selection_and_sex_channel_after_augmentation(sample, train):
    root, _, _, manifest, reference = sample
    dataset = BoneAgeDataset(manifest, root, reference, 0, train, 16)
    assert len(dataset) == 2
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["pid"] == ("002" if train else "001")
        assert item["image"].shape == (2, 16, 16)
        assert item["image"].dtype == torch.float32
        assert torch.all(item["image"][1] == (0 if train else 255))
        assert item["age"].shape == ()
        assert item["age"].dtype == torch.float32
        assert item["age"].item() == (240 if train else 0)


@pytest.mark.parametrize(
    "train,coin,expected", [(True, 0, False), (True, 1, True), (False, 0, True)]
)
def test_histogram_matching_probability_and_uint8(sample, train, coin, expected):
    root, _, _, manifest, reference = sample
    dataset = BoneAgeDataset(manifest, root, reference, 0, train, 8)
    seen = []

    def capture(*, image):
        seen.append(image.copy())
        return {"image": image}

    dataset.transforms = capture
    with (
        patch("bone_age_training.data.np.random.binomial", return_value=coin) as draw,
        patch(
            "bone_age_training.data.match_histograms", return_value=np.full((4, 8), 91.9)
        ) as match,
    ):
        dataset[0]
    assert match.called == expected
    assert draw.called == train
    if train:
        draw.assert_called_once_with(1, 0.5)
    assert seen[0].dtype == np.uint8
    assert seen[0].ndim == 2
    if expected:
        assert np.all(seen[0] == 91)


@pytest.mark.parametrize(
    "column,value",
    [("bone_age", -1), ("bone_age", 241), ("bone_age", np.inf), ("female", 2), ("fold", 0.5)],
)
def test_invalid_values(sample, column, value):
    root, _, _, manifest, _ = sample
    df = pd.read_csv(manifest)
    df[column] = value
    df.to_csv(manifest, index=False)
    with pytest.raises(ValueError):
        check_manifest(manifest, root)


def test_crop_preparation_allows_missing_generated_images(sample):
    root, _, _, manifest, _ = sample
    for p in (root / "cropped_train_plus_valid").iterdir():
        p.unlink()
    assert check_manifest(manifest, root, require_cropped=False)["patients"] == 2
    with pytest.raises(FileNotFoundError):
        check_manifest(manifest, root)


def test_prepare_rejects_subset_disagreement_and_existing_output(sample):
    root, labels, splits, manifest, _ = sample
    with pytest.raises(FileExistsError):
        prepare_manifest(labels, splits, manifest)
    df = pd.read_csv(labels, dtype={"pid": str})
    df["subset"] = ["valid", "valid"]
    df.to_csv(labels, index=False)
    with pytest.raises(ValueError, match="subset"):
        prepare_manifest(labels, splits, root / "new.csv")
