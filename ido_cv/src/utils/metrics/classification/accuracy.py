# -*- coding: utf-8 -*-
"""
Module implements functions for calculating metrics to compare
true labels and predicted labels using numpy
"""
import numpy as np
import torch
from . import BaseClassificationMetric


class Accuracy(BaseClassificationMetric):
    def __init__(self, mode: str, activation: str):
        # Assertions
        assert mode in ['binary', 'multi'], f'Wrong mode parameter: {mode}. ' \
            f'Should be binary of multi.'
        assert activation in ['sigmoid', 'softmax'], f'Wrong activation parameter: {activation}.'\
            f'Should be softmax or sigmoid.'

        self.activation = activation
        self.mode = mode
        self.metric_name = 'accuracy'

    def calculate_metric(
            self,
            trues: torch.Tensor,
            preds: torch.Tensor,
    ):
        trues_ = np.squeeze(trues.data.cpu().numpy(), axis=1).astype(np.uint8)
        # print(trues_.shape, trues_)

        if self.activation == 'softmax':
            preds_ = torch.softmax(preds, dim=-1).float()
            preds_ = np.argmax(preds_.data.cpu().numpy(), axis=-1).astype(np.uint8)
        elif self.activation == 'sigmoid':
            preds_ = torch.sigmoid(preds).float()
            preds_ = np.round(np.squeeze(preds_.data.cpu().numpy(), axis=1)).astype(np.uint8)
            # print(preds_.shape, preds_)
        else:
            raise ValueError(
                f'Activation for binary metric should be "sigmoid" or "softmax", '
                f'not {self.activation}'
            )

        metric = self.get_metric_value(trues=trues_, preds=preds_, metric_name=self.metric_name)

        return metric

    def __str__(self):
        return 'accuracy'
