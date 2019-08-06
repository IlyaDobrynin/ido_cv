# -*- coding: utf-8 -*-
"""
Module implements functions for test time augmentations (TTA)

"""
import numpy as np
import torch
from .model_utils import cuda


class TTAOp:
    """ Basic TTA class """
    
    def __init__(self, sigmoid=True, allocate_on='cpu'):
        self.sigmoid = sigmoid
        self.allocate_on = allocate_on

    def __call__(self, model, batch):
        with torch.no_grad():
            forwarded = cuda(torch.from_numpy(self.forward(batch.cpu().numpy())),
                             allocate_on=self.allocate_on)
        return self.backward(self.to_numpy(model(forwarded)))

    def forward(self, img):
        raise NotImplementedError

    def backward(self, img):
        raise NotImplementedError

    def to_numpy(self, batch):
        if self.sigmoid:
            batch = torch.sigmoid(batch)
        else:
            batch = torch.softmax(batch, dim=1)
        data = batch.data.cpu().numpy()
        return data


class BasicTTAOp(TTAOp):
    @staticmethod
    def op(img):
        raise NotImplementedError

    def forward(self, img):
        return self.op(img)

    def backward(self, img):
        return self.forward(img)


class Nothing(BasicTTAOp):
    @staticmethod
    def op(img):
        return img


class HFlip(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(np.flip(img, axis=2))


class VFlip(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(np.flip(img, axis=3))


class Transpose(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(img.transpose(0, 1, 3, 2))


def chain_op(data, operations):
    for op in operations:
        data = op.op(data)
    return data


class ChainedTTA(TTAOp):
    @property
    def operations(self):
        raise NotImplementedError

    def forward(self, img):
        return chain_op(img, self.operations)

    def backward(self, img):
        return chain_op(img, reversed(self.operations))


class HVFlip(ChainedTTA):
    @property
    def operations(self):
        return [HFlip, VFlip]


class TransposeHFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, HFlip]


class TransposeVFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, VFlip]


class TransposeHVFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, HFlip, VFlip]
