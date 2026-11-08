"""Image preprocessing, cropping and held-out evaluation."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.exposure import match_histograms
from transformers import AutoModel

from .assets import CROP_REVISION


def safe_image_path(root, relative):
    root = Path(root).resolve()
    relative = Path(str(relative))
    if relative.is_absolute() or not (root / relative).resolve().is_relative_to(root):
        raise ValueError(f"Image path must be relative to the data root: {relative}")
    return root / relative


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def cropped_image(image, model, device):
    prepared = model.preprocess(image)
    x = torch.from_numpy(np.ascontiguousarray(prepared)).float()[None, None].to(device)
    shape = torch.tensor([image.shape[:2]], device=device)
    with torch.inference_mode():
        coords = model(x, shape)[0].cpu().numpy()
    if not np.isfinite(coords).all():
        raise ValueError("Crop model returned nonfinite coordinates")
    x, y, width, height = coords.astype(int)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x + width), min(image.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Crop model returned an empty bounding box")
    return image[y0:y1, x0:x1]


def crop_manifest(manifest, data_root, device, overwrite=False):
    from .data import check_manifest

    check_manifest(manifest, data_root, require_cropped=False)
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    originals = {
        r["pid"]: r for r in rows if not r["imgfile"].startswith("cropped_train_plus_valid/")
    }
    pending = [
        (pid, row)
        for pid, row in originals.items()
        if overwrite or not (data_root / "cropped_train_plus_valid" / f"{pid}.png").is_file()
    ]
    if not pending:
        print("All crops already exist")
        return
    model = AutoModel.from_pretrained(
        "ianpan/bone-age-crop", revision=CROP_REVISION, trust_remote_code=True
    )
    model = model.eval().to(device)
    target = data_root / "cropped_train_plus_valid"
    target.mkdir(parents=True, exist_ok=True)
    for i, (pid, row) in enumerate(pending, 1):
        image = read_image(safe_image_path(data_root, row["imgfile"]))
        crop = cropped_image(image, model, device)
        if not cv2.imwrite(str(safe_image_path(target, f"{pid}.png")), crop):
            raise OSError(f"Could not write crop for {pid}")
        if i % 100 == 0:
            print(f"Cropped {i}/{len(pending)}", flush=True)
    check_manifest(manifest, data_root)


def evaluate(model_path, labels, data_root, image_dir, reference, crop, device, output):
    if output.exists():
        raise FileExistsError(output)
    with labels.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Evaluation labels are empty")
    required = {"pid", "female", "bone_age"}
    if not required <= rows[0].keys():
        raise ValueError(f"Evaluation table requires {sorted(required)}")
    if len({r["pid"] for r in rows}) != len(rows):
        raise ValueError("Evaluation IDs must be unique")
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True).eval().to(device)
    crop_model = (
        (
            AutoModel.from_pretrained(
                "ianpan/bone-age-crop", revision=CROP_REVISION, trust_remote_code=True
            )
            .eval()
            .to(device)
        )
        if crop
        else None
    )
    ref = read_image(reference) if reference else None
    predictions = []
    for row in rows:
        female, target = float(row["female"]), float(row["bone_age"])
        if female not in (0, 1) or not np.isfinite(target) or target < 0:
            raise ValueError("Invalid sex/age labels")
        relative = row.get("imgfile") or str(Path(image_dir) / f"{row['pid']}.png")
        image = read_image(safe_image_path(data_root, relative))
        if crop_model is not None:
            image = cropped_image(image, crop_model, device)
        if ref is not None:
            image = match_histograms(image, ref).astype("uint8")
        image = model.preprocess(image)
        x = torch.from_numpy(np.ascontiguousarray(image)).float()[None, None].to(device)
        with torch.inference_mode():
            age = model(x, torch.tensor([female], device=device)).item()
        predictions.append(
            {
                "pid": row["pid"],
                "bone_age": target,
                "prediction": age,
                "absolute_error": abs(age - target),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(
        json.dumps(
            {
                "images": len(predictions),
                "mae_months": float(np.mean([r["absolute_error"] for r in predictions])),
            },
            indent=2,
        )
    )
