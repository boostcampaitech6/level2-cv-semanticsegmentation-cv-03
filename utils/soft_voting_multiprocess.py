import os
import numpy as np
import pandas as pd
from util import IND2CLASS, encode_mask_to_rle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def main():
    save_csv_path = "/data/ephemeral/home/ensemble/deep_3_unet_base_fold5_0.3.csv"

    n_cpu = multiprocessing.cpu_count()

    logit_dir_paths = [
        "/data/ephemeral/home/ensemble/deeplabv3_efcb8/deeplabv3_efcb8_resolx2_logit_f1",
        "/data/ephemeral/home/ensemble/effb7/logit_f1",
        "/data/ephemeral/home/ensemble/deeplab_mobileone/logit_f1",
        "/data/ephemeral/home/ensemble/best_base_model/logit_f1",
        "/data/ephemeral/home/ensemble/best_base_model/logit_f2",
        "/data/ephemeral/home/ensemble/best_base_model/logit_f3",
        "/data/ephemeral/home/ensemble/best_base_model/logit_f4",
        "/data/ephemeral/home/ensemble/best_base_model/logit_f5",
    ]
    filenames_npz = sorted(os.listdir(logit_dir_paths[0]))
    
    weights = [1.5, 1.5, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0]
    all_thr = 0.3
    cls_thr = {
        'finger-1': 0, 
        'finger-2': 0,
        'finger-3': 0, 
        'finger-4': 0, 
        'finger-5': 0,
        'finger-6': 0, 
        'finger-7': 0, 
        'finger-8': 0, 
        'finger-9': 0, 
        'finger-10': 0,
        'finger-11': 0, 
        'finger-12': 0, 
        'finger-13': 0, 
        'finger-14': 0, 
        'finger-15': 0,
        'finger-16': 0, 
        'finger-17': 0, 
        'finger-18': 0, 
        'finger-19': 0, 
        'Trapezium': 0,
        'Trapezoid': 0, 
        'Capitate': 0, 
        'Hamate': 0, 
        'Scaphoid': 0, 
        'Lunate': 0,
        'Triquetrum': 0, 
        'Pisiform': 0, 
        'Radius': 0, 
        'Ulna': 0
    }

    multiple = 10000
    w_sum = sum(weights)
    
    result = multiple_process(logit_dir_paths, filenames_npz, weights, cls_thr, all_thr, multiple, w_sum, n_cpu)

    result.to_csv(save_csv_path, index=False)


def single_process(logit_dir_paths, fname, weights, cls_thr, all_thr, multiple, w_sum):
    global result
    filenames_png, classes, rles = [], [], []
    output = np.zeros((29, 2048, 2048), dtype=np.float32)

    for idx, d in enumerate(logit_dir_paths):
        fpath = os.path.join(d, fname)
        output += np.load(fpath)["output"].astype(np.float32) * weights[idx]

    if not all_thr :
        # class별 threshold 지정
        for c, _ in enumerate(output):
            output[c, :, :] = output[c, :, :] > cls_thr[IND2CLASS[c]] * multiple * w_sum
    else :
        output = output > all_thr * multiple * w_sum

    for c, segm in enumerate(output):
        rle = encode_mask_to_rle(segm)
        rles.append(rle)
        filenames_png.append(os.path.splitext(fname)[0] + ".png")
        classes.append(IND2CLASS[c])
    
    return pd.DataFrame({"image_name": filenames_png, "class": classes, "rle": rles})



def multiple_process(logit_dir_paths, filenames_npz, weights, cls_thr, all_thr, multiple, w_sum, n_cpu):
    cnt = len(filenames_npz)
    args = [(logit_dir_paths, fname, weights, cls_thr, all_thr, multiple, w_sum) for fname in filenames_npz]
    results = []
    with ProcessPoolExecutor(max_workers=n_cpu-1) as executor:
        for result in tqdm(executor.map(single_process, *zip(*args)), total=cnt):
            results.append(result)
    return pd.concat(results)
        

if __name__ == "__main__":
    main()