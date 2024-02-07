import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )

        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        Lovasz = self.lovasz_hinge(inputs, targets, per_image=False)
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
        self, inputs, targets, smooth=1, alpha=0.5, ce_ratio=0.5, eps=1e-9
    ):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = -(
            alpha
            * (
                (targets * torch.log(inputs))
                + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))
            )
        )
        weighted_ce = out.mean(-1)
        combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)

        return combo


def bce_with_logits_loss(inputs, targets):
    return nn.BCEWithLogitsLoss()(inputs, targets)


def dice_bce_loss(inputs, targets):
    return DiceBCELoss()(inputs, targets)


def dice_loss(inputs, targets):
    return DiceLoss()(inputs, targets)


def iou_loss(inputs, targets):
    return IoULoss()(inputs, targets)


def focal_loss(inputs, targets):
    return FocalLoss()(inputs, targets)


def tversky_loss(inputs, targets):
    return TverskyLoss()(inputs, targets)


def focal_tversky_loss(inputs, targets):
    return FocalTverskyLoss()(inputs, targets)


def lovasz_hinge_loss(inputs, targets):
    return LovaszHingeLoss()(inputs, targets)


def combo_loss(inputs, targets):
    return ComboLoss()(inputs, targets)
