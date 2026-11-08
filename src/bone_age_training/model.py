"""Two-head training model and expected-month loss."""

import torch
from timm import create_model
from torch import nn

from .config import TrainConfig
from .hf.modeling import GeM


def expected_age(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().softmax(dim=1)
    months = torch.arange(logits.shape[1], device=logits.device, dtype=probabilities.dtype)
    return (probabilities * months).sum(dim=1)


class BoneAgeNet(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.backbone = create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="",
            features_only=False,
            in_chans=2,
        )
        self.pooling = GeM(p=3, dim=2)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.ModuleList(
            [
                nn.Linear(self.backbone.num_features, 1),
                nn.Linear(self.backbone.num_features, config.num_classes),
            ]
        )

    def forward(self, image):
        features = self.dropout(self.pooling(self.backbone(image / 127.5 - 1)))
        return self.linear[0](features).squeeze(1), self.linear[1](features)


def double_mae(regression, logits, target):
    """Equal-weight regression and softmax expected-month absolute errors."""
    reg_loss = (regression.float() - target.float()).abs().mean()
    cls_loss = (expected_age(logits) - target.float()).abs().mean()
    return (reg_loss + cls_loss) / 2
