from base import BaseDataset
import albumentations as A

custom_transform = A.Compose([A.Resize(2048, 2048), A.Normalize()])


class CustomDataset(BaseDataset):
    def __init__(
        self,
        mmap_path,
        filenames,
        labelnames,
        hash_dict,
        labels,
        is_train=True,
    ):
        super().__init__(
            mmap_path,
            filenames,
            labelnames,
            hash_dict,
            labels,
            is_train=is_train,
        )
        self.transforms = custom_transform
