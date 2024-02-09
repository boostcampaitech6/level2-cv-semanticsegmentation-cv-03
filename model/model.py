import torch.nn as nn
import torchvision.models.segmentation as models
from base import BaseModel


class FCN_ResNet50(BaseModel):
    def __init__(self, num_classes=29):
        super().__init__()
        self.pretrained_model = models.fcn_resnet50(
            weights="DEFAULT",
        )

        self.pretrained_model.classifier[4] = nn.Conv2d(
            512, num_classes, kernel_size=1
        )

    def forward(self, x):
        return self.pretrained_model(x)
