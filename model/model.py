import torch.nn as nn
from base import BaseModel
from torchvision import models


class FCN_ResNet50(BaseModel):
    def __init__(self, num_classes=29):
        super().__init__()
        self.pretrained_model = models.segmentation.fcn_resnet50(
            pretrained=True
        )

        # output num_classes
        self.pretrained_model[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.pretrained_model(x)
