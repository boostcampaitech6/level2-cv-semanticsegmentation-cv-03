import os
from mmap_ninja import RaggedMmap
import cv2
import pickle
import numpy as np
import json
from utils.util import CLASS2IND, CLASSES


def generate_images(paths):
    for path in paths:
        yield cv2.imread(path)


if __name__ == "__main__":
    IMAGE_ROOT = "/data/ephemeral/home/datasets/train/DCM"
    LABEL_ROOT = "/data/ephemeral/home/datasets/train/outputs_json"
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    pngs = sorted(pngs)
    jsons = sorted(jsons)

    if "train_mmap" not in os.listdir("/data/ephemeral/home/datasets"):
        pngss = [os.path.join(IMAGE_ROOT, temp) for temp in pngs]
        RaggedMmap.from_generator(
            out_dir="/data/ephemeral/home/datasets/train_mmap",
            sample_generator=generate_images(pngss),
            batch_size=64,
            verbose=True,
        )
    if "hash.pickle" not in os.listdir("/data/ephemeral/home/datasets"):
        hash_dict = {}
        for i in range(len(pngs)):
            hash_dict[pngs[i]] = i

        with open("/data/ephemeral/home/datasets/data.pickle", "wb") as f:
            pickle.dump(hash_dict, f, pickle.HIGHEST_PROTOCOL)
        with open("/data/ephemeral/home/datasets/pngs.pickle", "wb") as f:
            pickle.dump(pngs, f, pickle.HIGHEST_PROTOCOL)
        with open("/data/ephemeral/home/datasets/jsons.pickle", "wb") as f:
            pickle.dump(jsons, f, pickle.HIGHEST_PROTOCOL)

    if "label.pickle" not in os.listdir("/data/ephemeral/home/datasets"):
        imgs = RaggedMmap("data/ephemeral/home/datasets/train_mmap")
        labels = {}

        for i in range(len(jsons)):
            image = imgs[i]
            label_path = os.path.join(LABEL_ROOT, jsons[i])

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

            labels[jsons[i]] = label

        with open("/ddata/ephemeral/home/datasets/label.pickle", "wb") as f:
            pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
