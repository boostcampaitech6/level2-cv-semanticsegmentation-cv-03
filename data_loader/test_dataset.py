import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A


class TestDataset(Dataset):
    def __init__(self, pngs, image_root, transforms=None):

        self.filenames = np.array(sorted(pngs))
        self.image_root = image_root
        self.transforms = (
            A.Compose(transforms) if transforms is not None else None
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        return image, image_name
