import torchvision.models.segmentation as models
from base import BaseModel


class FCN_ResNet50(BaseModel):
    def __init__(self, num_classes=29):
        super().__init__()
        self.pretrained_model = models.fcn_resnet50(
            weights="DEFAULT", num_classes=num_classes
        )

    def forward(self, x):
        return self.pretrained_model(x)
