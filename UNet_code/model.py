import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from loss import _lovasz_hinge as lovasz_loss

class SegmentationModel(nn.Module):
    def __init__(self, encoder, weights, class_weights = torch.tensor([0.875, 0.125])):
        super(SegmentationModel,self).__init__()

        self.cnn = smp.Unet(
            encoder_name = encoder,
            encoder_weights = weights,
            in_channels = 3,
            classes = 1,
            activation = None
        )

        self.class_weights = class_weights # Specify class weights as [ratio_class_0, ratio_class_1]

    def weighted_bce(self, logits, masks):
        weighted_bce = -masks * self.class_weights[1] * torch.log(logits) - (1 - masks) * self.class_weights[0] * torch.log(1 - logits)
        return torch.mean(weighted_bce)

    def forward(self, images, masks = None):
        # Predict
        logits = self.cnn(images)

        if masks != None: # Compute loss
            # loss1 = DiceLoss(mode = 'binary')(logits, masks)
            # loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            loss1 = lovasz_loss(logits, masks)
            loss2 = self.weighted_bce(logits, masks)
            return logits, loss1, loss2
        return logits