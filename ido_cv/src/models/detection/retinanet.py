# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from .fpn.fpn_custom import FPNFactory
from ..nn_blocks.se_blocks import SCSEBlock
from torch.autograd import Variable


class RetinaNet(nn.Module):
    num_anchors = 9
    
    def __init__(self, backbone, num_classes=10, se_block=False, residual=False):
        super(RetinaNet, self).__init__()
        self.fpn = FPNFactory(
            backbone=backbone, depth=5, pretrained='imagenet', unfreeze_encoder=True,
            fpn_features=256, include_last_conv_layers=False, num_input_channels=1,
            bn_type='default', conv_type='default', up_mode='nearest', residual=True,
            se_decoder=True
        )

        self.se_block = se_block
        self.residual = residual
        self.num_classes = num_classes
        self.loc_head = self._make_head()
        self.loc_last = nn.Conv2d(256, self.num_anchors * 4, kernel_size=3, stride=1, padding=1)
        
        self.cls_head = self._make_head()
        self.cls_last = nn.Conv2d(256, self.num_anchors * self.num_classes, kernel_size=3, stride=1,
                                  padding=1)
    
    def forward(self, x):
        fms = self.fpn(x)
        # print(len(fms))
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            
            if self.residual:
                loc_pred = loc_pred + fm
                cls_pred = cls_pred + fm
                
            loc_pred = self.loc_last(loc_pred)
            cls_pred = self.cls_last(cls_pred)
            
            # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            
            # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        out_loc = torch.cat(loc_preds, 1)
        out_cls = torch.cat(cls_preds, 1)
        return out_loc, out_cls
    
    def _make_head(self):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout2d(p=0.2))
        if self.se_block:
            layers.append(SCSEBlock(channel=256, reduction=16))
        return nn.Sequential(*layers)
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2, 3, 224, 224)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

# test()


if __name__ == '__main__':
    input_size = (3, 512, 512)
    model = RetinaNet()
    # print(model.state_dict())
    # model = backbone_factory.get_backbone(name=backbone, pretrained='imagenet')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    from torchsummary import summary
    summary(model, input_size=input_size)