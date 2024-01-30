import torch
import torch.nn as nn
import torch.nn.functional as F

"""Loss functions for semantic segmentation mainly used for binary segmentation problems"""

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #we should comment out if your model contains a sigmoid or equivalent activation layer
        """This is done to ensure that the predictions are in the range (0, 1), which is common in binary segmentation problems"""
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        """This is done to convert 2D tensors into 1D tensors, making it easier to calculate metrics and losses."""
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #Dice Coefficient calculation
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice #Dice loss is used for evaluating the similarity between predicted and ground truth segmentation masks.

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')   #computes the binary cross-entropy loss between the predicted and target tensors.
        Dice_BCE = BCE + dice_loss #used to balance the trade-off between localization accuracy (Dice) and pixel-wise classification accuracy (BCE).

        return Dice_BCE