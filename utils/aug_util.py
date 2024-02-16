import torch
import numpy as np


def mixup(images, masks, device, alpha=1.0):
    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = 1

    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(device, non_blocking=True)

    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]
    masks_a, masks_b = masks, masks[index]

    return mixed_images, masks_a, masks_b, lambda_


def mixup_loss(criterion, pred, labels_a, labels_b, lambda_):
    return lambda_ * criterion(pred, labels_a) + (1 - lambda_) * criterion(
        pred, labels_b
    )
