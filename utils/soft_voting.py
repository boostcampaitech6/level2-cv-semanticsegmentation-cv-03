import os
import numpy as np
import pandas as pd
from util import IND2CLASS, encode_mask_to_rle
from tqdm import tqdm

if __name__ == "__main__":

    save_csv_path = "/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03/result/ensemble.csv"

    logit_dir_paths = [
        "/data/ephemeral/home/datasets/logit_f1",
        "/data/ephemeral/home/datasets/logit_f2",
        "/data/ephemeral/home/datasets/logit_f3",
        "/data/ephemeral/home/datasets/logit_f4",
        "/data/ephemeral/home/datasets/logit_f5",
    ]
    filenames_npz = sorted(os.listdir(logit_dir_paths[0]))

    weights = [2.0, 1.0, 1.0, 1.0, 1.0]
    all_thr = 0.5
    cls_thr = {
        "finger-1": 0.5,
        "finger-2": 0.5,
        "finger-3": 0.5,
        "finger-4": 0.5,
        "finger-5": 0.5,
        "finger-6": 0.5,
        "finger-7": 0.5,
        "finger-8": 0.5,
        "finger-9": 0.5,
        "finger-10": 0.5,
        "finger-11": 0.5,
        "finger-12": 0.5,
        "finger-13": 0.5,
        "finger-14": 0.5,
        "finger-15": 0.5,
        "finger-16": 0.5,
        "finger-17": 0.5,
        "finger-18": 0.5,
        "finger-19": 0.5,
        "Trapezium": 0.5,
        "Trapezoid": 0.5,
        "Capitate": 0.5,
        "Hamate": 0.5,
        "Scaphoid": 0.5,
        "Lunate": 0.5,
        "Triquetrum": 0.5,
        "Pisiform": 0.5,
        "Radius": 0.5,
        "Ulna": 0.5,
    }

    multiple = 10000
    w_sum = sum(weights)

    rles = []
    filenames_png = []
    classes = []
    for fname in tqdm(filenames_npz, total=len(filenames_npz)):
        output = np.zeros((29, 2048, 2048), dtype=np.float32)

        for idx, d in enumerate(logit_dir_paths):
            fpath = os.path.join(d, fname)
            output += (
                np.load(fpath)["output"].astype(np.float32) * weights[idx]
            )

        # class별 threshold 지정
        # for c, _ in enumerate(output):
        #     output[c, :, :] = output[c, :, :] > cls_thr[IND2CLASS[c]] * multiple * w_sum

        output = output > all_thr * multiple * w_sum

        for c, segm in enumerate(output):
            rle = encode_mask_to_rle(segm)
            rles.append(rle)
            filenames_png.append(os.path.splitext(fname)[0] + ".png")
            classes.append(IND2CLASS[c])

    df = pd.DataFrame(
        {
            "image_name": filenames_png,
            "class": classes,
            "rle": rles,
        }
    )
    df.to_csv(save_csv_path, index=False)
