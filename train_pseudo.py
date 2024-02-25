import argparse
import collections
import os
import random
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
import torch.utils.data as module_data
from parse_config import ConfigParser
import model as module_arch
import data_loader as module_dataset
import trainer.loss as module_loss
import trainer.metric as module_metric
import torch.optim as module_optim
import torch.optim.lr_scheduler as module_lr
import trainer as module_trainer
from utils import prepare_device
import albumentations as A
import glob
from tqdm import tqdm
import torch.nn.functional as F

SEED = 42


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def predict_pseudo_label(config):
    cfg_path = config["path"]
    test_image_paths = glob.glob(
        os.path.join(cfg_path["test_root"], "*/*.png")
    )
    test_image_paths = np.array(sorted(test_image_paths))

    save_test_mask_root = cfg_path["test_mask_root"]
    if not os.path.exists(save_test_mask_root):
        os.mkdir(save_test_mask_root)

    test_tf_list = []
    for tf in config["test_transforms"]:
        test_tf_list.append(
            getattr(A, tf["name"])(*tf["args"], **tf["kwargs"])
        )

    test_dataset = config.init_obj(
        "test_dataset",
        module_dataset,
        image_paths=test_image_paths,
        transforms=test_tf_list,
    )
    test_data_loader = config.init_obj(
        "test_data_loader", module_data, test_dataset
    )

    threshold = config["threshold"]["pred_thr"]

    model = config.init_obj("arch", module_arch)
    model.load_state_dict(
        torch.load(cfg_path["inference_model_path"])["state_dict"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for images, image_paths in tqdm(test_data_loader, desc="(Inference)"):
            images = images.to(device)

            outputs = model(images)
            outputs = F.interpolate(
                outputs, size=(2048, 2048), mode="bilinear"
            )
            outputs = torch.sigmoid(outputs)
            outputs = (
                (outputs > threshold).detach().cpu().numpy().astype(np.uint8)
            )

            for output, image_path in zip(outputs, image_paths):
                save_file_name = (
                    os.path.splitext(os.path.basename(image_path))[0] + ".bin"
                )
                packed_data = np.packbits(output.reshape(-1))
                packed_data.tofile(
                    os.path.join(save_test_mask_root, save_file_name)
                )


def main(config):
    predict_pseudo_label(config)

    cfg_path = config["path"]
    image_paths = glob.glob(os.path.join(cfg_path["train_root"], "*/*.png"))
    test_image_paths = glob.glob(
        os.path.join(cfg_path["test_root"], "*/*.png")
    )

    image_paths = np.array(sorted(image_paths))
    test_image_paths = sorted(test_image_paths)

    mask_root = cfg_path["mask_root"]
    mask_paths = [
        os.path.join(
            mask_root, os.path.splitext(os.path.basename(fpath))[0] + ".bin"
        )
        for fpath in image_paths
    ]
    mask_paths = np.array(mask_paths)

    test_mask_root = cfg_path["test_mask_root"]
    test_mask_paths = [
        os.path.join(
            test_mask_root,
            os.path.splitext(os.path.basename(fpath))[0] + ".bin",
        )
        for fpath in test_image_paths
    ]

    train_tf_list, test_tf_list = [], []
    for tf in config["train_transforms"]:
        train_tf_list.append(
            getattr(A, tf["name"])(*tf["args"], **tf["kwargs"])
        )
    for tf in config["test_transforms"]:
        test_tf_list.append(
            getattr(A, tf["name"])(*tf["args"], **tf["kwargs"])
        )

    groups = [os.path.dirname(fpath) for fpath in image_paths]
    gkf = GroupKFold(n_splits=config["kfold"]["n_splits"])
    fold = 1

    for i, (x, y) in enumerate(
        gkf.split(image_paths, mask_paths, groups), start=1
    ):
        if fold == i:
            train_image_paths = list(image_paths[x])
            train_mask_paths = list(mask_paths[x])
            valid_image_paths = list(image_paths[y])
            valid_mask_paths = list(mask_paths[y])
            break

    train_image_paths.extend(test_image_paths)
    train_mask_paths.extend(test_mask_paths)

    train_dataset = config.init_obj(
        "train_dataset",
        module_dataset,
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        transforms=train_tf_list,
    )

    valid_dataset = config.init_obj(
        "valid_dataset",
        module_dataset,
        image_paths=valid_image_paths,
        mask_paths=valid_mask_paths,
        transforms=test_tf_list,
    )

    train_data_loader = config.init_obj(
        "train_data_loader", module_data, train_dataset
    )
    valid_data_loader = config.init_obj(
        "valid_data_loader", module_data, valid_dataset
    )

    model = config.init_obj("arch", module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", module_optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", module_lr, optimizer)

    train_kwargs = {
        "model": model,
        "criterion": criterion,
        "metrics": metrics,
        "optimizer": optimizer,
        "config": config,
        "device": device,
        "train_data_loader": train_data_loader,
        "valid_data_loader": valid_data_loader,
        "lr_scheduler": lr_scheduler,
        "fold": fold,
    }
    trainer = config.init_obj("trainer", module_trainer, **train_kwargs)

    trainer.train()


if __name__ == "__main__":
    set_seeds(SEED)

    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03/config_pseudo.json",
        type=str,
        help="config file path",
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
        CustomArgs(["-e", "--epoch"], type=int, target="trainer;epochs"),
        CustomArgs(["-n", "--name"], type=str, target="wandb;exp_name"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
