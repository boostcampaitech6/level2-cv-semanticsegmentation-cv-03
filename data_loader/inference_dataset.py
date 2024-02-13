import os
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2


class InferenceDataset(Dataset):
    def __init__(self, pngs, image_root, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.image_root = image_root
        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        return image, image_name
