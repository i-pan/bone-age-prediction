"""Export three last-checkpoint classification heads without uploading."""

import json
import shutil

import torch

from .assets import sha256
from .hf.configuration import BoneAgeConfig
from .hf.modeling import BoneAgeEnsembleModel


def classification_state(checkpoint):
    """Keep the trained backbone/GeM and rename head 1 to the inference linear head."""
    state = checkpoint["state_dict"]
    result = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    for name in ("weight", "bias"):
        result[f"linear.{name}"] = result.pop(f"linear.1.{name}")
        result.pop(f"linear.0.{name}")
    return result


def export(checkpoints, reference, output):
    if len(checkpoints) != 3:
        raise ValueError("Supply folds 0, 1 and 2 in that order")
    if output.exists():
        raise FileExistsError(output)
    if not reference.is_file():
        raise FileNotFoundError(reference)
    paths = [p.resolve() for p in checkpoints]
    if len(set(paths)) != 3:
        raise ValueError("Checkpoint paths must be distinct")
    payloads = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    recipes = [p["hyper_parameters"]["config"] for p in payloads]
    architecture = ("backbone", "dropout", "num_classes", "image_size")
    if any(any(c[k] != recipes[0][k] for k in architecture) for c in recipes):
        raise ValueError("Ensemble checkpoint architectures/preprocessing must match")
    if any(p.get("training_fold") != i for i, p in enumerate(payloads)):
        raise ValueError("Checkpoint fold metadata must identify folds 0, 1, 2 in order")
    identity = payloads[0].get("training_data")
    if not identity or any(p.get("training_data") != identity for p in payloads):
        raise ValueError(
            "Ensemble members must share their training manifest and histogram reference"
        )
    if identity["reference_sha256"] != sha256(reference):
        raise ValueError("Export reference differs from the reference used for training")
    if recipes[0]["image_size"] != 512:
        raise ValueError("Release export preprocessing expects 512-pixel training images")
    states = [classification_state(p) for p in payloads]
    c = recipes[0]
    feature_dim = states[0]["linear.weight"].shape[1]
    cfg = BoneAgeConfig(
        backbone=c["backbone"],
        dropout=c["dropout"],
        num_classes=c["num_classes"],
        feature_dim=feature_dim,
    )
    BoneAgeConfig.register_for_auto_class()
    BoneAgeEnsembleModel.register_for_auto_class("AutoModel")
    model = BoneAgeEnsembleModel(cfg)
    for fold, state in enumerate(states):
        getattr(model, f"net{fold}").load_state_dict(state, strict=True)
    provenance = [
        {
            "fold": i,
            "checkpoint": paths[i].name,
            "sha256": sha256(paths[i]),
            "training_config": recipes[i],
            "training_data": identity,
        }
        for i in range(3)
    ]
    model.save_pretrained(output, safe_serialization=True)
    shutil.copyfile(reference, output / "ref_img.png")
    (output / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Bone age ensemble\n\nThree-fold classification-head ensemble, "
        "exported by bone-age-training. Predictions are in months.\n\n"
        "Load with `AutoModel.from_pretrained(path, trust_remote_code=True)`. "
        "Preprocess a grayscale image with `model.preprocess`, then pass its "
        "NCHW float tensor and a binary `female` tensor to the model. "
        "Crop and histogram-match against `ref_img.png` before preprocessing.\n"
    )
    print(f"Exported three model members to {output}")
