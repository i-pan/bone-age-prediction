"""Explicit settings from the bone-age release training run."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    backbone: str = "convnextv2_tiny"
    pretrained: bool = True
    image_size: int = 512
    dropout: float = 0.1
    num_classes: int = 240
    batch_size: int = 32
    steps_per_epoch: int = 2000
    epochs: int = 10
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    div_factor: float = 100.0
    final_div_factor: float = 1000.0
    workers: int = 16
    seed: int = 42
    precision: str = "16-mixed"
    sync_batchnorm: bool = True

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text()))

    def validate(self):
        for name in ("image_size", "num_classes", "batch_size", "steps_per_epoch", "epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_classes != 240:
            raise ValueError("Bone-age release training uses exactly 240 month classes")
        if self.workers < 0 or not 0 < self.warmup_fraction < 1:
            raise ValueError("Invalid workers or warmup_fraction")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.div_factor <= 0 or self.final_div_factor <= 0:
            raise ValueError("Learning-rate divisors must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)
