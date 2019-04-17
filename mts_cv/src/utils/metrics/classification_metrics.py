# -*- coding: utf-8 -*-
"""
Module implements functions for calculating metrics to compare
true labels and predicted labels using numpy
"""
import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(y_true, y_pred):
    """ Function returns mean accuracy score for input batch of images

    :return:
    """
    acc = accuracy_score(y_true, y_pred.round())
    return acc

def multi_accuracy(y_true, y_pred):
    # y_pred = np.argmax(y_pred, axis=-1)
    # print(y_true)
    # print(y_pred)
    acc = accuracy_score(y_true, y_pred.round())
    return acc