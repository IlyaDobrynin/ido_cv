from torch import nn
from torch.nn import functional as F
from ido_cv.src.utils.loss.loss_utils import lovasz_softmax


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