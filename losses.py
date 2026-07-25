import torch
import torch.nn.functional as F

def dice_coeff(pred, target, eps=1e-4):
    # pred/target: (N,1,H,W), pred in [0,1]
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()

def dice_loss_from_logits(logits, target):
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    return 1.0 - dice_coeff(pred, target)

def bce_dice_loss(logits, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dloss = dice_loss_from_logits(logits, target)
    return bce_weight * bce + (1 - bce_weight) * dloss