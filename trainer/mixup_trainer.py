import numpy as np
from torch.cuda.amp import autocast
from trainer import Trainer
from utils import mixup, mixup_loss
from tqdm import tqdm


class MixUpTrainer(Trainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        train_data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        fold=1,
        mixup_prob=0.3,
    ):
        super().__init__(
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            train_data_loader,
            valid_data_loader,
            lr_scheduler,
            fold,
        )
        self.mixup_prob = mixup_prob

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.to(self.device)
        self.model.train()
        self.train_metrics.reset()

        for _, (images, masks) in enumerate(
            tqdm(self.train_data_loader, desc=f"[Epoch {epoch} (Train)]")
        ):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # mixup 적용 확률
            apply_mixup = np.random.rand() < self.mixup_prob

            if apply_mixup:
                mixed_images, masks_a, masks_b, lambda_ = mixup(
                    images, masks, self.device, alpha=1.0
                )

            self.optimizer.zero_grad()

            with autocast():
                if apply_mixup:
                    outputs = self.model(mixed_images)
                    loss = mixup_loss(
                        self.criterion, outputs, masks_a, masks_b, lambda_
                    )
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.train_metrics.update("loss", loss.item())

            # outputs = torch.sigmoid(outputs)
            # outputs = (outputs > self.pred_thr).detach().cpu()
            # masks = masks.detach().cpu()

            # for met in self.metrics:
            #     self.train_metrics.update(met.__name__, met(outputs, masks))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        log_str = [
            f"\033[94m{key:>10}: {value:.6f} | \033[0m"
            for key, value in [
                ("train_loss", log["loss"]),
                ("valid_loss", val_log["loss"]),
                ("valid_dice", val_log["dice_coef"]),
            ]
        ]
        log_str = "".join(log_str)
        print(log_str)

        if self.lr_scheduler is not None:
            if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.lr_scheduler.step(log[self.mnt_metric])
            else:
                self.lr_scheduler.step()

        return log
