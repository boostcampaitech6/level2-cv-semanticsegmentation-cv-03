import argparse
import collections
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
from utils import IND2CLASS, encode_mask_to_rle
from parse_config import ConfigParser
from tqdm import tqdm


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
    image_root = config["path"]["test_path"]
    threshold = config["threshold"]["pred_thr"]
    save_csv_path = config["path"]["save_csv_path"]
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    test_tf_list = []
    for tf in config["test_transforms"]:
        test_tf_list.append(
            getattr(A, tf["name"])(*tf["args"], **tf["kwargs"])
        )

    test_dataset = config.init_obj(
        "test_dataset",
        module_dataset,
        pngs=pngs,
        image_root=image_root,
        transforms=test_tf_list,
    )
    test_data_loader = config.init_obj(
        "test_data_loader", module_data, test_dataset
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.load(config["path"]["inference_model_path"])["state_dict"]
    )

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(
            enumerate(test_data_loader), total=len(test_data_loader)
        ):
            images = images.cuda()
            outputs = model(images)

            outputs = F.interpolate(
                outputs, size=(2048, 2048), mode="bilinear"
            )
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > threshold).detach().cpu().numpy()
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )

    df.to_csv(save_csv_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03/config.json",
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
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["-m", "--model_path"],
            type=str,
            target="path;inference_model_path",
        ),
        CustomArgs(
            ["-o", "--save_path"], type=str, target="path;save_csv_path"
        ),
        CustomArgs(
            ["-t", "--pred_thr"], type=int, target="threshold;pred_thr"
        ),
    ]
    config = ConfigParser.from_args(args, options, mode="inference")

    main(config)
