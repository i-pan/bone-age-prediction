"""Two-process CPU validation/checkpoint lifecycle probe; no training updates."""

import json
import sys
from datetime import timedelta
from pathlib import Path

import conftest  # noqa: F401  Registers the tiny test backbone, fixes thread count/offline mode.
import lightning as L
import pandas as pd
import torch
from lightning.pytorch.strategies import DDPStrategy
from test_training_export import fixture_data
from torch import nn

from bone_age_training.config import TrainConfig
from bone_age_training.training import BoneAgeDataModule, BoneAgeModule


class KnownPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, image):
        n = image.shape[0]
        return self.offset.expand(n), self.offset.expand(n, 240)


class SaveResults(L.Callback):
    def __init__(self, root):
        self.root = root

    def on_validation_end(self, trainer, module):
        # All ranks must participate in saving, even though only global rank zero writes.
        trainer.save_checkpoint(self.root / "distributed.ckpt")
        result = {k: float(v) for k, v in trainer.callback_metrics.items()}
        result["world_size"] = trainer.world_size
        (self.root / f"rank-{trainer.global_rank}.json").write_text(json.dumps(result))


def main(root):
    fixture_data(root)
    frame = pd.read_csv(root / "manifest.csv", dtype={"pid": str})
    frame["fold"] = 0
    # a,a,b,c,b,c -> ranks see unequal target means (60 vs 100).
    frame = frame.iloc[[0, 3, 1, 2, 4, 5]]
    frame.to_csv(root / "manifest.csv", index=False)
    config = TrainConfig(
        backbone="bone_age_testnet", pretrained=False, image_size=16, batch_size=1, workers=0
    )
    module = BoneAgeModule(config.to_dict())
    module.model = KnownPredictor()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(
            start_method="fork", process_group_backend="gloo", timeout=timedelta(seconds=20)
        ),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
        callbacks=[SaveResults(root)],
    )
    trainer.validate(
        module,
        datamodule=BoneAgeDataModule(config, root / "manifest.csv", root, root / "ref.png", 0),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
