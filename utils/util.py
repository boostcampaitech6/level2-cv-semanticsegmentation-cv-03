import json
import torch
import pandas as pd
from pathlib import Path
from collections import OrderedDict


CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, classes=None):
        self.keys = keys
        self.classes = classes
        self.dices_per_class = None

        self._data = pd.DataFrame(
            index=keys, columns=["total", "counts", "average"]
        )

        self.reset()

    def reset(self):
        self.dices_per_class = None

        for key in self._data.index:
            for col in self._data.columns:
                self._data.loc[key, col] = 0

            if key == "dice_coef":
                self._data.loc[key, "total"] = []

    def update(self, key, value):
        if key == "dice_coef":
            self._data.loc[key, "total"].append(value)
            self._data.loc[key, "counts"] += 1
        else:
            self._data.loc[key, "total"] += value
            self._data.loc[key, "counts"] += 1

    def avg_dice(self, key="dice_coef"):
        self._data.loc[key, "total"] = torch.cat(
            self._data.loc[key, "total"], 0
        )
        self.dices_per_class = torch.mean(self._data.loc[key, "total"], 0)

        return torch.mean(self.dices_per_class).item()

    def result(self):
        for key in self.keys:
            if key == "dice_coef":
                self._data.loc[key, "average"] = self.avg_dice(key)
            else:
                self._data.loc[key, "average"] = (
                    self._data.loc[key, "total"]
                    / self._data.loc[key, "counts"]
                )

        concat_df = self._data

        if self.classes is not None:
            dice_df = pd.DataFrame(
                index=self.classes, columns=["total", "counts", "average"]
            )

            for key in dice_df.index:
                for col in dice_df.columns:
                    dice_df.loc[key, col] = 0

            for key, value in zip(self.classes, self.dices_per_class):
                dice_df.loc[key, "average"] = value.item()

            concat_df = pd.concat([self._data, dice_df])

        return dict(concat_df.average)
