from base import BaseDataset
import albumentations as A


class CustomDataset(BaseDataset):
    def __init__(
        self,
        mmap_path,
        filenames,
        labelnames,
        hash_dict,
        labels,
        is_train=True,
        transforms=A.Compose([A.Resize(2048, 2048), A.Normalize()]),
    ):
        super().__init__(
            mmap_path,
            filenames,
            labelnames,
            hash_dict,
            labels,
            is_train=is_train,
        )
        self.transforms = transforms
