import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from utils import CLASSES


class MaskLoadDataset(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        transforms=None,
    ):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = (
            A.Compose(transforms) if transforms is not None else None
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        mask_path = self.mask_paths[item]

        image = cv2.imread(image_path)

        mask_shape = (image.shape[0], image.shape[1], len(CLASSES))

        loaded_mask = np.fromfile(mask_path, dtype=np.uint8)
        mask = np.unpackbits(loaded_mask).reshape(mask_shape)

        if self.transforms is not None:
            inputs = {"image": image, "mask": mask}
            result = self.transforms(**inputs)

            image = result["image"]
            mask = result["mask"]

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask
