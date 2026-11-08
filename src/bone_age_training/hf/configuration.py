from transformers import PretrainedConfig


class BoneAgeConfig(PretrainedConfig):
    model_type = "bone_age"

    def __init__(
        self,
        backbone="convnextv2_tiny",
        feature_dim=768,
        dropout=0.1,
        num_classes=240,
        in_chans=2,
        num_models=3,
        **kwargs,
    ):
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_models = num_models
        super().__init__(**kwargs)
