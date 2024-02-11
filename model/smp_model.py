import segmentation_models_pytorch as smp
from base import BaseModel


class Unet(BaseModel):
    def __init__(
        self,
        num_classes=29,
        encoder_name="resnet34",
        encoder_weights="imagenet",
    ):
        super().__init__()
        self.pretrained_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
        )

    def forward(self, x):
        return self.pretrained_model(x)


class Linknet(BaseModel):
    def __init__(
        self,
        num_classes=29,
        encoder_name="resnet34",
        encoder_weights="imagenet",
    ):
        super().__init__()
        self.pretrained_model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
        )

    def forward(self, x):
        return self.pretrained_model(x)


class FPN(BaseModel):
    def __init__(
        self,
        num_classes=29,
        encoder_name="resnet34",
        encoder_weights="imagenet",
    ):
        super().__init__()
        self.pretrained_model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
        )

    def forward(self, x):
        return self.pretrained_model(x)


class PSPNet(BaseModel):
    def __init__(
        self,
        num_classes=29,
        encoder_name="resnet34",
        encoder_weights="imagenet",
    ):
        super().__init__()
        self.pretrained_model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
        )

    def forward(self, x):
        return self.pretrained_model(x)


class PAN(BaseModel):
    def __init__(
        self,
        num_classes=29,
        encoder_name="resnet34",
        encoder_weights="imagenet",
    ):
        super().__init__()
        self.pretrained_model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
        )

    def forward(self, x):
        return self.pretrained_model(x)
