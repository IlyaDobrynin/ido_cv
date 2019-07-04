import torch
from torch import nn


class MultiRobustFocalLoss2d(nn.Module):
    def __init__(
            self,
            gamma: int = 2,
            size_average: bool = True,
            class_weight: list = None
    ):
        super().__init__()
        self.class_weight = class_weight
        self.gamma = gamma
        self.size_average = size_average

    def make_loss(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor
    ):
        target = trues.view(-1, 1).long()
        if self.class_weight is None:
            self.class_weight = [1] * 2  # [0.5, 0.5]

        B, C, H, W = preds.size()
        if self.class_weight is None:
            self.class_weight = [1] * C  # [1/C]*C

        logit = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)
        prob = torch.softmax(logit, dim=1)
        select = torch.Tensor(len(prob), C).zero_().cuda()
        select.scatter_(1, target, 1.)

        self.class_weight = torch.Tensor(self.class_weight).cuda().view(-1, 1)
        self.class_weight = torch.gather(self.class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)
        batch_loss = - self.class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss

    def __call__(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor
    ):
        loss = self.make_loss(
            preds=preds,
            trues=trues
        )

        return loss
