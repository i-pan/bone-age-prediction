"""Command-line tools for data, training, evaluation and local export."""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="Training companion for ianpan/bone-age")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare", help="Join source labels with release fold assignments")
    q.add_argument("--labels", type=Path, required=True)
    q.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("reference", help="Download the published histogram reference image")
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("crop", help="Generate crops using ianpan/bone-age-crop")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--data-root", type=Path, required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--overwrite", action="store_true")
    q = sub.add_parser("check-data", help="Validate images, labels and patient fold consistency")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--data-root", type=Path, required=True)
    q = sub.add_parser("train", help="Train one independently held-out fold")
    q.add_argument("--config", type=Path, default=Path("configs/release.json"))
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--data-root", type=Path, required=True)
    q.add_argument("--reference", type=Path, required=True)
    q.add_argument("--fold", type=int, required=True, choices=range(5))
    q.add_argument("--devices", type=int, default=2)
    q.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--resume", type=Path)
    for field in ("steps-per-epoch", "epochs", "batch-size", "image-size", "workers"):
        q.add_argument("--" + field, type=int)
    q.add_argument("--no-pretrained", action="store_true")
    q.add_argument(
        "--limit-val-batches", type=int, help="Optional number of validation batches for smoke runs"
    )
    q = sub.add_parser("export", help="Write a local three-model Hugging Face package")
    q.add_argument("--checkpoints", nargs=3, type=Path, required=True)
    q.add_argument("--reference", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("evaluate", help="Evaluate a Hugging Face ensemble on held-out images")
    q.add_argument("--model", required=True)
    q.add_argument("--labels", type=Path, required=True)
    q.add_argument("--data-root", type=Path, required=True)
    q.add_argument("--image-dir", default="test")
    q.add_argument("--reference", type=Path)
    q.add_argument("--crop", action="store_true")
    q.add_argument("--device", default="cpu")
    q.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_manifest

        # Preserve user label files; normalize only an ephemeral input table.
        with args.labels.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        if "imgfile" not in fields:
            fields.append("imgfile")
        for row in rows:
            if not row.get("imgfile"):
                if row.get("subset") not in ("train", "valid"):
                    raise ValueError("Provide subset=train/valid or an explicit imgfile")
                row["imgfile"] = f"{row['subset']}/{row['pid']}.png"
        with tempfile.TemporaryDirectory() as temp:
            labels = Path(temp) / "labels.csv"
            with labels.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            prepare_manifest(labels, args.splits, args.output)
        print(f"Wrote {args.output}")
    elif args.command == "reference":
        if args.output.exists():
            raise FileExistsError(args.output)
        from huggingface_hub import hf_hub_download

        from .assets import REFERENCE_SHA256, RELEASE_REVISION, sha256

        path = hf_hub_download("ianpan/bone-age", "ref_img.png", revision=RELEASE_REVISION)
        if sha256(path) != REFERENCE_SHA256:
            raise ValueError("Histogram reference does not match the release asset")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, args.output)
        print(f"Saved {args.output}")
    elif args.command == "check-data":
        from .data import check_manifest

        print(json.dumps(check_manifest(args.manifest, args.data_root), indent=2))
    elif args.command == "crop":
        from .inference import crop_manifest

        crop_manifest(args.manifest, args.data_root, args.device, args.overwrite)
    elif args.command == "train":
        from .config import TrainConfig
        from .training import train

        config = TrainConfig.load(args.config)
        for name in ("steps_per_epoch", "epochs", "batch_size", "image_size", "workers"):
            if getattr(args, name) is not None:
                setattr(config, name, getattr(args, name))
        if args.no_pretrained:
            config.pretrained = False
        train(
            config,
            args.manifest,
            args.data_root,
            args.reference,
            args.fold,
            args.devices,
            args.accelerator,
            args.output,
            args.limit_val_batches if args.limit_val_batches is not None else 1.0,
            args.resume,
        )
    elif args.command == "export":
        from .export import export

        export(args.checkpoints, args.reference, args.output)
    elif args.command == "evaluate":
        from .inference import evaluate

        evaluate(
            args.model,
            args.labels,
            args.data_root,
            args.image_dir,
            args.reference,
            args.crop,
            args.device,
            args.output,
        )


if __name__ == "__main__":
    main()
