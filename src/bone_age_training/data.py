"""Patient-safe manifests and the original bone-age image preprocessing."""

from pathlib import Path, PureWindowsPath

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.exposure import match_histograms
from torch.utils.data import Dataset


def _read(path: Path, columns: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"pid": str}, keep_default_na=False)
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df.pid.str.strip().eq("").any():
        raise ValueError("Manifest must contain nonempty patient IDs")
    return df


def _path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative image path: {value!r}")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError(f"Image path must stay within data root: {value!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError(f"Image path escapes data root: {value!r}")
    return resolved


def _validate(df: pd.DataFrame) -> None:
    for column in ("bone_age", "female", "fold"):
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must be finite")
    if not df.bone_age.between(0, 240).all():
        raise ValueError("bone_age must be between 0 and 240 months")
    if not df.female.isin([0, 1]).all():
        raise ValueError("female must be binary (0 or 1)")
    if not ((df.fold.between(0, 4)) & (df.fold % 1 == 0)).all():
        raise ValueError("fold must be an integer from 0 to 4")
    if df.subset.astype(str).str.strip().eq("").any():
        raise ValueError("subset must be nonempty")
    for column in ("fold", "subset", "female", "bone_age"):
        if (df.groupby("pid")[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"Patient leakage or inconsistent {column} metadata")
    paths = [str(_path(p, Path("/"))) for p in df.imgfile]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths")


def prepare_manifest(labels_csv: Path, splits_csv: Path, output_csv: Path) -> None:
    """Join normalized labels to preserved splits and add one cropped view."""
    if Path(output_csv).exists():
        raise FileExistsError(output_csv)
    labels = _read(labels_csv, {"pid", "bone_age", "female", "imgfile"})
    splits = _read(splits_csv, {"pid", "fold", "subset"})
    if labels.pid.duplicated().any() or splits.pid.duplicated().any():
        raise ValueError("Labels and splits must have unique patient IDs")
    if set(labels.pid) != set(splits.pid):
        raise ValueError("Labels and splits must have exact patient coverage")
    if "subset" in labels:
        expected = splits.set_index("pid").subset
        if not labels.subset.eq(labels.pid.map(expected)).all():
            raise ValueError("Label subset disagrees with preserved split subset")
    df = labels[["pid", "bone_age", "female", "imgfile"]].merge(
        splits[["pid", "fold", "subset"]], on="pid", validate="one_to_one"
    )
    for pid in df.pid:
        if "/" in pid or "\\" in pid or pid in {".", ".."}:
            raise ValueError(f"Unsafe patient ID: {pid!r}")
    cropped = df.copy()
    cropped["imgfile"] = "cropped_train_plus_valid/" + cropped.pid + ".png"
    result = pd.concat([df, cropped], ignore_index=True)
    _validate(result)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)


def _manifest(manifest: Path) -> pd.DataFrame:
    df = _read(manifest, {"pid", "bone_age", "female", "imgfile", "fold", "subset"})
    _validate(df)
    return df


def check_manifest(manifest: Path, data_root: Path, require_cropped: bool = True) -> dict:
    """Reject leakage, unsafe paths, missing images and missing cropped views."""
    df = _manifest(manifest)
    paths = [_path(p, Path(data_root)) for p in df.imgfile]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate resolved image paths")
    for relative, path in zip(df.imgfile, paths, strict=True):
        is_crop = relative.startswith("cropped_train_plus_valid/")
        if not path.is_file() and (require_cropped or not is_crop):
            raise FileNotFoundError(f"Missing image: {path}")
    if require_cropped:
        for pid, group in df.groupby("pid"):
            crop = f"cropped_train_plus_valid/{pid}.png"
            if len(group) != 2 or crop not in set(group.imgfile):
                raise ValueError(f"Patient {pid} requires original and cropped images")
    return {
        "rows": len(df),
        "patients": df.pid.nunique(),
        "folds": {int(k): int(v) for k, v in df.groupby("fold").pid.nunique().items()},
    }


def build_transforms(train: bool, image_size: int = 512) -> A.Compose:
    """Reproduce the Albumentations 1.4.24 source configuration."""
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    transforms = [
        A.LongestMaxSize(max_size=image_size, p=1),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1,
        ),
    ]
    if train:
        transforms += [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.SomeOf(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.2,
                        scale_limit=0,
                        rotate_limit=0,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1,
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=0.2,
                        rotate_limit=0,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1,
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=0,
                        rotate_limit=30,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1,
                    ),
                    A.GaussianBlur(p=1),
                    A.GaussNoise(var_limit=None, std_range=(0.2, 0.44), noise_scale_factor=1, p=1),
                    A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0, p=1),
                    A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=0.3, p=1),
                ],
                n=3,
                p=0.9,
                replace=False,
            ),
        ]
    return A.Compose(transforms)


class BoneAgeDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        reference: Path,
        fold: int,
        train: bool,
        image_size: int = 512,
    ):
        df = _manifest(manifest)
        self.data_root = Path(data_root)
        # Validate every path before filtering, including held-out patients.
        paths = [_path(p, self.data_root) for p in df.imgfile]
        if len(set(paths)) != len(paths):
            raise ValueError("Duplicate resolved image paths")
        if fold not in set(df.fold):
            raise ValueError(f"Unknown fold: {fold}")
        self.df = df.loc[df.fold.ne(fold) if train else df.fold.eq(fold)].reset_index(drop=True)
        self.train = train
        self.transforms = build_transforms(train, image_size)
        self.reference = self._load(Path(reference))

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Missing or unreadable image: {path}")
        return image

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image = self._load(_path(row.imgfile, self.data_root))
        if not self.train or np.random.binomial(1, 0.5):
            image = match_histograms(image, self.reference).astype(np.uint8)
        image = self.transforms(image=image)["image"]
        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()
        sex = torch.full_like(image, 255.0 * row.female)
        return {
            "image": torch.cat([image, sex]),
            "age": torch.tensor(float(row.bone_age), dtype=torch.float32),
            "pid": str(row.pid),
        }
