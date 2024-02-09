import torch


def dice_coef(outputs, masks):
    y_true_f = masks.flatten(2)
    y_pred_f = outputs.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2.0 * intersection + eps) / (
        torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
    )
