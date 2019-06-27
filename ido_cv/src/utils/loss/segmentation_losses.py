import torch
from torch import nn
from torch.nn import functional as F
from ..loss_utils import dice_coef
from ..loss_utils import jaccard_coef
from ..loss_utils import lovasz_hinge
from ..loss_utils import lovasz_softmax
from ..loss_utils import get_weight


class BinaryBceMetric(nn.Module):
    """ Loss defined as (1 - alpha) * BCE - alpha * SoftJaccard
    """

    def __init__(self, metric=None, weight_type=None, is_average=False, alpha=0.3,
                 per_image: bool = None, ignore: int = None):
        super().__init__()
        self.metric = metric
        self.weight_type = weight_type
        self.is_average = is_average
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()

        if self.metric == 'lovasz':
            if per_image is None:
                raise ValueError(
                    f"If metric parameter is lovasz, parameter per_image should be set."
                )
            self.per_image = per_image
            self.ignore = ignore

    def __call__(self, preds, trues):
        metric_target = (trues == 1).float()
        metric_output = torch.sigmoid(preds)

        # Weight estimation
        if self.weight_type:
            weights = get_weight(trues=trues, weight_type=self.weight_type)
        else:
            weights = None

        bce_loss = self.bce_loss(preds, trues)
        if self.metric:
            if self.metric == 'jaccard':
                metric_coef = jaccard_coef(metric_target, metric_output, weight=weights)
            elif self.metric == 'dice':
                metric_coef = dice_coef(metric_target, metric_output, weight=weights)
            elif self.metric == 'lovasz':
                metric_coef = lovasz_hinge(
                    metric_output, metric_target, self.per_image, self.ignore
                )
            else:
                raise NotImplementedError(
                    f"Metric {self.metric} doesn't implemented. "
                    f"Should be 'jaccard', 'dice', 'lovasz' or None."
                )
            loss = self.alpha * bce_loss - (1 - self.alpha) * torch.log(metric_coef)
        else:
            loss = bce_loss
        return loss


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def __call__(self, preds, targets, class_weight=None, type='sigmoid'):
        target = targets.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = torch.sigmoid(preds)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = preds.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = torch.softmax(logit, 1)
            select = torch.Tensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        else:
            raise ValueError(
                f'Wrong type of activation function: {type}. Should be "sigmoid" or "softmax".'
            )

        class_weight = torch.Tensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


class MultiLovasz(nn.Module):

    def __init__(self, ignore=0):
        super().__init__()
        if ignore is not None:
            self.ignore = ignore
        else:
            self.ignore = None

    def __call__(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        loss = lovasz_softmax(probas=outputs, labels=targets, ignore=self.ignore)
        return loss


class MultiBceMetric(nn.Module):
    def __init__(self,  metric: str, alpha: float = 0.3, class_weights: list = None,
                 num_classes=2, ignore_class: int = None):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.metric = metric
        self.ignore_class = ignore_class
        if class_weights is None:
            class_weights = [1] * num_classes
            self.class_weights = torch.Tensor(class_weights)
        else:
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"Length od class weights should be the same as num classes. "
                    f"Give: num_classes = {num_classes}, "
                    f"len(class_weights) = {len(class_weights)}."
                )
            self.class_weights = torch.Tensor(class_weights)

    def __call__(self, preds: torch.Tensor, trues: torch.Tensor):
        metric_output = torch.softmax(preds, dim=1)
        loss = 0
        for i in range(metric_output.shape[1]):
            if i == self.ignore_class:
                continue

            cls_weight = self.class_weights[i]
            metric_target_cls = (trues == i).float()
            metric_output_cls = metric_output[:, i, ...].unsqueeze(1)

            # self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            self.bce_loss = nn.BCEWithLogitsLoss()
            bce_loss = self.bce_loss(
                metric_output_cls,
                metric_target_cls
            )
            if self.metric:
                if self.metric == 'jaccard':
                    metric_coef = jaccard_coef(metric_target_cls, metric_output_cls)
                elif self.metric == 'dice':
                    metric_coef = dice_coef(metric_target_cls, metric_output_cls)
                else:
                    raise NotImplementedError(
                       f"Metric {self.metric} doesn't implemented. "
                       f"Variants: 'jaccard;, 'dice', None")
                loss += (1 - self.alpha) * bce_loss - \
                        self.alpha * torch.log(metric_coef) * cls_weight
            else:
                loss += bce_loss

            loss *= cls_weight

        if self.ignore_class is not None:
            ignore = 1
        else:
            ignore = 0

        loss = loss / (metric_output.shape[1] - ignore)
        return loss


