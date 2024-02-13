import argparse
import collections
import os
import random
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data as module_data
import data_loader as module_dataset
import model as module_arch
from parse_config import ConfigParser
from tqdm import tqdm

SEED = 42
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
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


def main(config):
    image_root = config["path"]["test_path"]
    threshold = config["threshold"]["pred_thr"]
    save_csv_path = config["path"]["save_csv_path"]
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    test_dataset = config.init_obj(
        "test_dataset",
        module_dataset,
        pngs=pngs,
        image_root=image_root,
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

    set_seeds(SEED)
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
    warnings.simplefilter(action="ignore", category=FutureWarning)
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
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
    config = ConfigParser.from_args(args, options)

    main(config)
