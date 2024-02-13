# python native
import os
import json

# external library
import cv2
import numpy as np
from mmap_ninja.ragged import RaggedMmap
import albumentations as A

# torch
import torch
from torch.utils.data import Dataset
from utils.util import CLASS2IND,CLASSES



class XRayDataset(Dataset):
    def __init__(
        self,
        mmap_path,
        filenames,
        labelnames,
        hash_dict,
        label_root,
        is_train=True,
    ):

        self.label_root = label_root
        self.mmap = RaggedMmap(mmap_path)
        self.hash_dict = hash_dict
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = A.Compose([A.Resize(1024, 1024), A.Normalize()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        # image_path = os.path.join(self.image_root, image_name)

        image = self.mmap[self.hash_dict[image_name]]

        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)

        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = (
                {"image": image, "mask": label}
                if self.is_train
                else {"image": image}
            )
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class XRayInferenceDataset(Dataset):
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