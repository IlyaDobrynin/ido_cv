import torch
from torch import nn
from torch.nn import functional as F


class CTCLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, outputs, targets):
        target_images = targets[0]
        target_text = targets[1]
        target_lengths = targets[2]

        preds = F.log_softmax(outputs, dim=2)
        batch_size = target_images.size(0)
        preds_size = torch.full(
            (batch_size,),
            outputs.size(0),
            dtype=torch.int32
        )
        loss_f = nn.CTCLoss()
        loss = loss_f(preds, target_text, preds_size, target_lengths)

        return loss