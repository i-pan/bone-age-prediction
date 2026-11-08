from typing import Optional

import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from timm import create_model
from transformers import PreTrainedModel

from .configuration import BoneAgeConfig


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6, dim: int = 2, flatten: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        assert dim in {2, 3}, f"dim must be one of [2, 3], not {dim}"
        self.dim = dim
        if self.dim == 2:
            self.func = F.adaptive_avg_pool2d
        elif self.dim == 3:
            self.func = F.adaptive_avg_pool3d
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assumes x.shape is (n, c, [t], h, w)
        x = self.func(x.clamp(min=self.eps).pow(self.p), output_size=1).pow(1.0 / self.p)
        return self.flatten(x)


class BoneAgeModel(nn.Module):
    def __init__(self, backbone, feature_dim=768, dropout=0.1, num_classes=240, in_chans=2):
        super().__init__()
        self.backbone = create_model(
            model_name=backbone,
            pretrained=False,
            num_classes=0,
            global_pool="",
            features_only=False,
            in_chans=in_chans,
        )
        self.pooling = GeM(p=3, dim=2)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(feature_dim, num_classes)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # [0, 255] -> [-1, 1]
        mini, maxi = 0.0, 255.0
        x = (x - mini) / (maxi - mini)
        x = (x - 0.5) * 2.0
        return x

    def forward(
        self, x: torch.Tensor, female: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor:
        assert x.size(0) == female.size(0), (
            f"x.size(0) [{x.size(0)}] must equal female.size(0) [{female.size(0)}]"
        )
        female_ch = torch.zeros_like(x).to(x.device)
        female_ch[female.bool()] = 255.0
        x = torch.cat([x, female_ch], dim=1)
        x = self.normalize(x)
        features = self.pooling(self.backbone(x))
        logits = self.linear(features)
        if return_logits:
            return logits
        out = (logits.softmax(1) * torch.arange(logits.size(1)).to(logits.device)).sum(1)
        return out


class BoneAgeEnsembleModel(PreTrainedModel):
    config_class = BoneAgeConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_models = config.num_models
        for i in range(self.num_models):
            setattr(
                self,
                f"net{i}",
                BoneAgeModel(
                    config.backbone,
                    config.feature_dim,
                    config.dropout,
                    config.num_classes,
                    config.in_chans,
                ),
            )

    @staticmethod
    def load_image_from_dicom(path: str) -> Optional[NDArray]:
        try:
            from pydicom import dcmread
            from pydicom.pixels import apply_voi_lut
        except ModuleNotFoundError:
            print("`pydicom` is not installed, returning None ...")
            return None
        dicom = dcmread(path)
        arr = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr

        arr = arr - arr.min()
        arr = arr / arr.max()
        arr = (arr * 255).astype("uint8")
        return arr

    @staticmethod
    def preprocess(x: NDArray) -> NDArray:
        x = A.LongestMaxSize(max_size=512, p=1)(image=x)["image"]
        x = A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, p=1)(image=x)["image"]
        return x

    def forward(
        self, x: torch.Tensor, female: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor:
        out = []
        for i in range(self.num_models):
            model = getattr(self, f"net{i}")
            out.append(model(x, female, return_logits))
        return torch.stack(out).mean(0)
