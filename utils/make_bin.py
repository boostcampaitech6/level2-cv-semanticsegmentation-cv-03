import os
import json
import glob
import cv2
import numpy as np
from tqdm import tqdm
from util import CLASS2IND

if __name__ == "__main__":
    LABEL_ROOT = "/data/ephemeral/home/datasets/train/outputs_json"
    label_paths = glob.glob(os.path.join(LABEL_ROOT, "*/*.json"))
    label_paths = sorted(label_paths)

    SAVE_ROOT = "/data/ephemeral/home/datasets/mask_bin"
    if not os.path.exists(SAVE_ROOT):
        os.mkdir(SAVE_ROOT)

    for label_path in tqdm(label_paths, total=len(label_paths)):
        mask_shape = (2048, 2048, 29)
        mask = np.zeros(mask_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            class_mask = np.zeros((2048, 2048), dtype=np.uint8)
            cv2.fillPoly(class_mask, [points], 1)
            mask[..., class_ind] = class_mask

        save_file_name = (
            os.path.splitext(os.path.basename(label_path))[0] + ".bin"
        )
        packed_data = np.packbits(mask.reshape(-1))
        packed_data.tofile(os.path.join(SAVE_ROOT, save_file_name))
