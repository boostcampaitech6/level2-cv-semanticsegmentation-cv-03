# external library
from mmap_ninja.ragged import RaggedMmap

# torch
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        mmap_path,
        filenames,
        labelnames,
        hash_dict,
        labels,
        is_train=True,
    ):

        self.mmap = RaggedMmap(mmap_path)
        self.hash_dict = hash_dict
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.labels = labels
        self.transforms = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        # image_path = os.path.join(self.image_root, image_name)
        image = self.mmap[self.hash_dict[image_name]]

        label_name = self.labelnames[item]
        label = self.labels[label_name]

        # Transform
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
