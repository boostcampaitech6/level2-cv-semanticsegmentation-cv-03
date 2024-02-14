import argparse
import collections
import os
import random
import pickle
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
    # load files
    cfg_path = config["path"]
    with open(cfg_path["image_name_pickle_path"], "rb") as f:
        filenames = np.array(pickle.load(f))
    with open(cfg_path["label_name_pickle_path"], "rb") as f:
        labelnames = np.array(pickle.load(f))
    with open(cfg_path["image_dict_pickle_path"], "rb") as f:
        hash_dict = pickle.load(f)

    # group k-fold
    groups = [os.path.dirname(fname) for fname in filenames]
    ys = [0 for _ in filenames]
    gkf = GroupKFold(n_splits=config["kfold"]["n_splits"])

    for fold, (x, y) in enumerate(gkf.split(filenames, ys, groups), start=1):
        train_filenames = list(filenames[x])
        train_labelnames = list(labelnames[x])
        valid_filenames = list(filenames[y])
        valid_labelnames = list(labelnames[y])

        train_dataset = config.init_obj(
            "train_dataset",
            module_dataset,
            mmap_path=cfg_path["mmap_path"],
            filenames=train_filenames,
            labelnames=train_labelnames,
            hash_dict=hash_dict,
            label_root=cfg_path["label_path"],
        )

        valid_dataset = config.init_obj(
            "valid_dataset",
            module_dataset,
            mmap_path=cfg_path["mmap_path"],
            filenames=valid_filenames,
            hash_dict=hash_dict,
            labelnames=valid_labelnames,
            label_root=cfg_path["label_path"],
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
        trainable_params = filter(
            lambda p: p.requires_grad, model.parameters()
        )
        optimizer = config.init_obj(
            "optimizer", module_optim, trainable_params
        )
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

        if fold == config["kfold"]["n_iter"]:
            break


if __name__ == "__main__":
    set_seeds(SEED)

    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03/config.json",
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
