# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..loss_utils import one_hot_embedding


class FocalLoss(nn.Module):
    def __init__(
            self,
            num_classes=10
    ):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)  # [N,21]
        # print('loss.t:', t)
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = w * ((1 - pt) ** gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, reduction='sum')

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)
        t = t[:, 1:]
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    def focal_loss_alt1(self, x, y):
        import torch
        focusing_param = 2
        balance_param = 0.25
        # cross_entropy = F.cross_entropy(x, y)
        # cross_entropy_log = torch.log(cross_entropy)
        # t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)  # [N,21]
        # print('loss.t:', t)
        # t = t[:, 1:]  # exclude background
        # t = Variable(y).cuda()  # [N,20]
        # t = t.long()
        logpt = - F.cross_entropy(x, y)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** focusing_param) * logpt

        balanced_focal_loss = balance_param * focal_loss

        return balanced_focal_loss

    def make_loss(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor
    ):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        loc_preds, cls_preds = preds[0], preds[1]
        loc_targets, cls_targets = targets[0], targets[1]
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.float().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        # cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        # cls_loss = self.focal_loss_alt1(masked_cls_preds, cls_targets[pos_neg])
        # print("Loss:", type(cls_loss))
        # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        loss = (loc_loss + cls_loss) / num_pos

        out_dict = dict(
            loss=loss,
            loc_loss=loc_loss / num_pos,
            cls_loss=cls_loss / num_pos
        )
        return loss
