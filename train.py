import argparse
import collections
import os
import random
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from data_loader.dataset import XRayDataset
import pickle
from sklearn.model_selection import GroupKFold


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


def main(config):
    # logger = config.get_logger("train")

    # 필요한 파일들 load
    with open("/data/ephemeral/home/datasets/pngs.pickle", "rb") as f:
        _filenames = np.array(pickle.load(f))
    with open("/data/ephemeral/home/datasets/jsons.pickle", "rb") as f:
        _labelnames = np.array(pickle.load(f))
    with open("/data/ephemeral/home/datasets/data.pickle", "rb") as f:
        hash_dict = pickle.load(f)

    LABEL_ROOT = "/data/ephemeral/home/datasets/train/outputs_json"
    MMAP_PATH = "/data/ephemeral/home/datasets/train_mmap"
    # setup data_loader instances

    # group-kfold 구현
    groups = [os.path.dirname(fname) for fname in _filenames]
    ys = [0 for fname in _filenames]
    gkf = GroupKFold(n_splits=5)

    for _, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
        train_filenames = list(_filenames[x])
        train_labelnames = list(_labelnames[x])
        valid_filenames = list(_filenames[y])
        valid_labelnames = list(_labelnames[y])

        train_dataset = XRayDataset(
            mmap_path=MMAP_PATH,
            filenames=train_filenames,
            labelnames=train_labelnames,
            hash_dict=hash_dict,
            label_root=LABEL_ROOT,
            is_train=True,
        )
        valid_dataset = XRayDataset(
            mmap_path=MMAP_PATH,
            filenames=valid_filenames,
            hash_dict=hash_dict,
            labelnames=valid_labelnames,
            label_root=LABEL_ROOT,
            is_train=False,
        )

        train_data_loader = config.init_obj(
            "train_data_loader", torch.utils.data, train_dataset
        )
        valid_data_loader = config.init_obj(
            "valid_data_loader", torch.utils.data, valid_dataset
        )
        # build model architecture, then print to console
        model = config.init_obj("arch", module_arch)
        # logger.info(model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config["n_gpu"])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config["loss"])
        metrics = [getattr(module_metric, met) for met in config["metrics"]]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(
            lambda p: p.requires_grad, model.parameters()
        )
        optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
        lr_scheduler = config.init_obj(
            "lr_scheduler", torch.optim.lr_scheduler, optimizer
        )

        trainer = Trainer(
            model,
            criterion,
            metrics,
            optimizer,
            config=config,
            device=device,
            data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
            lr_scheduler=lr_scheduler,
        )

        trainer.train()
        break


if __name__ == "__main__":
    set_seeds(SEED)

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
            ["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"
        ),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target="data_loader;args;batch_size",
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
