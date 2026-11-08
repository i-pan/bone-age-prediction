import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from timm.models import register_model
from torch import nn

torch.set_num_threads(1)


@register_model
def bone_age_testnet(pretrained=False, in_chans=2, **kwargs):
    class TinyBackbone(nn.Module):
        num_features = 4

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_chans, self.num_features, 3, padding=1)

        def forward(self, x):
            return self.conv(x).relu()

    return TinyBackbone()
