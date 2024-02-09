import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from base import BaseTrainer
from utils import MetricTracker
from tqdm import tqdm


class Trainer(BaseTrainer):
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
    ):
        super().__init__(model, optimizer, config)
        self.device = device

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = GradScaler()

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler

        self.metrics = metrics
        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metrics]
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metrics]
        )

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

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.train_metrics.update("loss", loss.item())
            for met in self.metrics:
                self.train_metrics.update(met.__name__, met(outputs, masks))

        log = self.train_metrics.result()

        for key, value in log.items():
            print(f"    {key:15s}: {value} |")

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

            for key, value in val_log.items():
                print(f"    {key:15s}: {value} |")

        if self.lr_scheduler is not None:
            if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.lr_scheduler.step(val_log["loss"])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for _, (images, masks) in enumerate(
                tqdm(self.valid_data_loader, desc=f"[Epoch {epoch} (Valid)]")
            ):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with autocast():
                    outputs = self.model(images)["out"]

                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)

                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(
                            outputs, size=(mask_h, mask_w), mode="bilinear"
                        )

                    loss = self.criterion(outputs, masks)

                self.valid_metrics.update("loss", loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(
                        met.__name__, met(outputs, masks)
                    )

        return self.valid_metrics.result()
