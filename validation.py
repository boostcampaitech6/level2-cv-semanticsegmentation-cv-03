import argparse
import collections
import json
import pickle
import os
import random
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data as module_data
import data_loader as module_dataset
import model as module_arch
import albumentations as A
from utils import IND2CLASS, encode_mask_to_rle, CLASSES
from parse_config import ConfigParser
from tqdm import tqdm
from sklearn.model_selection import GroupKFold


def dice_coef(outputs, masks):
    y_true_f = masks.flatten(2)
    y_pred_f = outputs.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    eps = 0.0001
    return (2.0 * intersection + eps) / (
        torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
    )


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(config):
    set_seeds()
    molel_name = config["path"]["model_path"].split('/')[-2]
    save_csv_path = config["path"]["save_csv_path"]
    thresholds = config["thresholds"]
    cfg_path = config["path"]
    with open(cfg_path["image_name_pickle_path"], "rb") as f:
        filenames = np.array(pickle.load(f))
    with open(cfg_path["label_name_pickle_path"], "rb") as f:
        labelnames = np.array(pickle.load(f))
    with open(cfg_path["image_dict_pickle_path"], "rb") as f:
        hash_dict = pickle.load(f)
    
    valid_tf_list = []
    for tf in config["valid_transforms"]:
        valid_tf_list.append(
            getattr(A, tf["name"])(*tf["args"], **tf["kwargs"])
        )

    # group k-fold
    groups = [os.path.dirname(fname) for fname in filenames]
    ys = [0 for _ in filenames]
    gkf = GroupKFold(n_splits=config["kfold"]["n_splits"])
    for fold, (x, y) in enumerate(gkf.split(filenames, ys, groups), start=1):
        if fold != config["kfold"]["fold"]: continue
        valid_filenames = list(filenames[y])
        valid_labelnames = list(labelnames[y])
        valid_dataset = config.init_obj(
            "valid_dataset",
            module_dataset,
            filenames=valid_filenames,
            labelnames=valid_labelnames,
            hash_dict=hash_dict,
            mmap_path=cfg_path["mmap_path"],
            label_root=cfg_path["label_path"],
            transforms=valid_tf_list,
        )
        
    valid_data_loader = config.init_obj(
        "valid_data_loader", module_data, valid_dataset
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.load(config["path"]["model_path"])["state_dict"] 
    )

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for threshold in thresholds:
            dices = []
            rles = []
            filename_and_class = []
            for step, (images, masks, image_names) in tqdm(enumerate(valid_data_loader), total=len(valid_data_loader)):
                images, masks = images.cuda(), masks.cuda()
                outputs = model(images)

                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")

                outputs = torch.sigmoid(outputs)
                outputs = (outputs > threshold).detach().cpu()
                masks = masks.detach().cpu()

                dice = dice_coef(outputs, masks)
                dices.append(dice)

                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name.replace('_','-')}")
            
            dices = torch.cat(dices, 0)
            dices_per_class = torch.mean(dices, 0)
            dice_str = [
                f"{d.item():.4f}"
                for c, d in zip(CLASSES, dices_per_class)
            ]
            dice_str = "\n".join(dice_str)
            avg_dice = torch.mean(dices_per_class).item()
            print(dice_str)
            print(f'{avg_dice:.4f}')

            classes, filename = zip(*[x.split("_") for x in filename_and_class])
            image_name = [os.path.basename(f) for f in filename]
            df = pd.DataFrame(
                {
                    "image_name": image_name,
                    "class": classes,
                    "rle": rles,
                }
            )
            df.to_csv(f'{save_csv_path}/{molel_name}_{threshold}.csv', index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03/config_inference.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    config = ConfigParser.from_args(args, mode="inference")
    main(config)
