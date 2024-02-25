import sys
import os
import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as module_data
import data_loader as module_dataset
import model as module_arch
import albumentations as A
from tqdm import tqdm

if True:
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from parse_config import ConfigParser


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

    save_logit_root = config["path"]["save_logit_root"]
    if not os.path.exists(save_logit_root):
        os.mkdir(save_logit_root)

    image_root = config["path"]["test_path"]
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

    model = config.init_obj("arch", module_arch)
    model.load_state_dict(
        torch.load(config["path"]["inference_model_path"])["state_dict"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _, (images, image_names) in tqdm(
            enumerate(test_data_loader), total=len(test_data_loader)
        ):
            images = images.to(device)
            outputs = model(images)

            outputs = F.interpolate(
                outputs, size=(2048, 2048), mode="bilinear"
            )
            outputs = torch.sigmoid(outputs).detach().cpu().numpy()
            comp_outputs = (outputs * 10000).astype(np.uint16)

            for i in range(len(image_names)):
                np.savez_compressed(
                    os.path.join(
                        save_logit_root,
                        os.path.splitext(os.path.basename(image_names[i]))[0]
                        + ".npz",
                    ),
                    output=comp_outputs[i],
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03/config_logit.json",
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
    config = ConfigParser.from_args(args, mode="inference")

    main(config)
