import os
from mmap_ninja import RaggedMmap
import cv2
import pickle


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
