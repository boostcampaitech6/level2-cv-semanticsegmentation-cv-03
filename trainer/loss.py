import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import losses


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, masks, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        masks = masks.view(-1)

        intersection = (outputs * masks).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            outputs.sum() + masks.sum() + smooth
        )
        BCE = F.binary_cross_entropy(outputs, masks, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, masks, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        masks = masks.view(-1)

        intersection = (outputs * masks).sum()
        dice = (2.0 * intersection + smooth) / (
            outputs.sum() + masks.sum() + smooth
        )

        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, outputs, masks, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        masks = masks.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (outputs * masks).sum()
        total = (outputs + masks).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, outputs, masks, alpha=0.8, gamma=2, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        masks = masks.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(outputs, masks, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, outputs, masks, smooth=1, alpha=0.5, beta=0.5):

        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        masks = masks.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (outputs * masks).sum()
        FP = ((1 - masks) * outputs).sum()
        FN = (masks * (1 - outputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, outputs, masks, smooth=1, alpha=0.5, beta=0.5, gamma=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        masks = masks.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (outputs * masks).sum()
        FP = ((1 - masks) * outputs).sum()
        FN = (masks * (1 - outputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, outputs, masks):
        outputs = F.sigmoid(outputs)
        Lovasz = self.lovasz_hinge(outputs, masks, per_image=False)
        return Lovasz

    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = labels != ignore
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
        """
        if per_image:
            losses = [
                self.lovasz_hinge_flat(
                    *self.flatten_binary_scores(
                        log.unsqueeze(0), lab.unsqueeze(0), ignore
                    )
                )
                for log, lab in zip(logits, labels)
            ]
            loss = sum(losses) / (len(losses) + 0.000001)
        else:
            loss = self.lovasz_hinge_flat(
                *self.flatten_binary_scores(logits, labels, ignore)
            )
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
        logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    # =====
    # Multi-class Lovasz loss
    # =====

    def lovasz_softmax(
        self, probas, labels, classes="present", per_image=False, ignore=None
    ):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        if per_image:
            losses = [
                self.lovasz_softmax_flat(
                    *self.flatten_probas(
                        prob.unsqueeze(0), lab.unsqueeze(0), ignore
                    ),
                    classes=classes
                )
                for prob, lab in zip(probas, labels)
            ]
            loss = sum(losses) / (len(losses) + 0.000001)
        else:
            loss = self.lovasz_softmax_flat(
                *self.flatten_probas(probas, labels, ignore), classes=classes
            )
        return loss

    def lovasz_softmax_flat(self, probas, labels, classes="present"):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.0
        C = probas.size(1)
        losses = []
        class_to_sum = (
            list(range(C)) if classes in ["all", "present"] else classes
        )
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError(
                        "Sigmoid output possible only with 1 class"
                    )
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(
                torch.dot(errors_sorted, self.lovasz_grad(fg_sorted))
            )
        return sum(losses) / (len(losses) + 0.000001)


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(
        self, outputs, masks, smooth=1, alpha=0.5, ce_ratio=0.5, eps=1e-9
    ):

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        masks = masks.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (outputs * masks).sum()
        dice = (2.0 * intersection + smooth) / (
            outputs.sum() + masks.sum() + smooth
        )

        outputs = torch.clamp(outputs, eps, 1.0 - eps)
        out = -(
            alpha
            * (
                (masks * torch.log(outputs))
                + ((1 - alpha) * (1.0 - masks) * torch.log(1.0 - outputs))
            )
        )
        weighted_ce = out.mean(-1)
        combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)

        return combo


class ComboLoss2(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss2, self).__init__()

    def forward(self, outputs, masks):

        bce = bce_with_logits_loss(outputs, masks)
        dice = dice_loss(outputs, masks)

        return bce * 0.5 + dice * 0.5


def bce_with_logits_loss(outputs, masks):
    return nn.BCEWithLogitsLoss()(outputs, masks)


def dice_bce_loss(outputs, masks):
    return DiceBCELoss()(outputs, masks)


def dice_loss(outputs, masks):
    return DiceLoss()(outputs, masks)


def iou_loss(outputs, masks):
    return IoULoss()(outputs, masks)


def focal_loss(outputs, masks):
    return FocalLoss()(outputs, masks)


def tversky_loss(outputs, masks):
    return TverskyLoss()(outputs, masks)


def focal_tversky_loss(outputs, masks):
    return FocalTverskyLoss()(outputs, masks)


def lovasz_hinge_loss(outputs, masks):
    return LovaszHingeLoss()(outputs, masks)


def combo_loss(outputs, masks):
    return ComboLoss()(outputs, masks)


def combo_loss2(outputs, masks):
    return ComboLoss2()(outputs, masks)


def jaccard_loss(outputs, masks):
    loss = losses.JaccardLoss(mode="multilabel")
    return loss(outputs, masks)


def smp_dice_loss(outputs, masks):
    loss = losses.DiceLoss(mode="multilabel")
    return loss(outputs, masks)


def combo_loss3(outputs, masks):
    return (
        jaccard_loss(outputs, masks) * 0.5
        + bce_with_logits_loss(outputs, masks) * 0.5
    )


def combo_loss4(outputs, masks):
    return (
        jaccard_loss(outputs, masks) * 0.4
        + bce_with_logits_loss(outputs, masks) * 0.3
        + smp_dice_loss(outputs, masks) * 0.3
    )


def combo_loss5(outputs, masks):
    return (
        jaccard_loss(outputs, masks) * 0.5
        + smp_dice_loss(outputs, masks) * 0.5
    )
