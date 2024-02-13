import torch
from abc import abstractmethod
from numpy import inf
import os
from logger import WandbLogger


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, optimizer, config, fold=1):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.fold = fold

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")
        self.logger = WandbLogger(self.config, self.model, self.fold)
        self.save_dir = config.save_dir

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"lr": self.optimizer.param_groups[0]["lr"]}
            log.update(result)
            self.logger.log_info(log, epoch)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min"
                        and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max"
                        and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    print(
                        f"Warning: Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled."
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print(
                        f"Validation performance didn't improve for {self.early_stop} epochs. Training stops."
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
            if best:
                self._save_best_checkpoint(epoch)

        self.logger.finish()

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        file_path = os.path.join(
            self.save_dir, f"f{self.fold}_epoch_{epoch}.pth"
        )
        torch.save(state, file_path)
        print(f"Saving checkpoint: {file_path} ...")

        # self.logger.log_artifact(file_path)
        # print(f"Saving wandb artifact: {file_path} ...")

    def _save_best_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        best_path = os.path.join(self.save_dir, f"f{self.fold}_best.pth")
        torch.save(state, best_path)
        print(f"Saving current best epoch {epoch}: {best_path} ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print(f"Loading checkpoint: {resume_path} ...")

        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            print(
                "Warning: Architecture configuration given in config file is different from that of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"]["type"]
            != self.config["optimizer"]["type"]
        ):
            print(
                "Warning: Optimizer type given in config file is different from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )
