from torch.nn import BCEWithLogitsLoss
from monai.losses import DiceLoss
import torch


class DiceBCELoss(torch.nn.Module):
    def __init__(self, sigmoid=True,dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.diceloss = DiceLoss(sigmoid=sigmoid)
        self.bceloss = BCEWithLogitsLoss()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight


    def forward(self,pred,target):
        dice = self.diceloss(pred,target)
        bce = self.bceloss(pred,target)

        return self.dice_weight*dice + self.bce_weight*bce