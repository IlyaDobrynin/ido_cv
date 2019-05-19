# -*- coding: utf-8 -*-
"""
    Some helper functions
"""
import os
import math
import pandas as pd
import cv2
import torch
import torch.nn as nn


def get_mean_and_std(dataset, max_load=10000):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i in range(N):
        print(i)
        im,_,_ = dataset.load(1)
        for j in range(3):
            mean[j] += im[:,j,:,:].mean()
            std[j] += im[:,j,:,:].std()
    mean.div_(N)
    std.div_(N)
    return mean, std


def mask_select(input, mask, dim=0):
    '''Select tensor rows/cols using a mask tensor.

    Args:
      input: (tensor) input tensor, sized [N,M].
      mask: (tensor) mask tensor, sized [N,] or [M,].
      dim: (tensor) mask dim.

    Returns:
      (tensor) selected rows/cols.

    Example:
    >>> a = torch.randn(4,2)
    >>> a
    -0.3462 -0.6930
     0.4560 -0.7459
    -0.1289 -0.9955
     1.7454  1.9787
    [torch.FloatTensor of size 4x2]
    >>> i = a[:,0] > 0
    >>> i
    0
    1
    0
    1
    [torch.ByteTensor of size 4]
    >>> masked_select(a, i, 0)
    0.4560 -0.7459
    1.7454  1.9787
    [torch.FloatTensor of size 2x2]
    '''
    index = mask.nonzero().squeeze(1)
    return input.index_select(dim, index)


def meshgrid(x, y, step=1, row_major=True):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0, x, step=step).float()
    b = torch.arange(0, y, step=step).float()
    
    xx = a.repeat(y).view(-1,1)
    # print(xx)
    yy = b.view(-1,1).repeat(1, x).view(-1,1)
    # print(yy)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a+1], 1)
    return torch.cat([a-b/2,a+b/2], 1)


def box_iou(box1, box2, order='xywh'):
    '''Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    intersection = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = intersection / (area1[:,None] + area2 - intersection)
    return iou


def box_iou_alt(box1, box2):
    """ Alternative function to get intersection over union metric for two bboxes

    :param box1: bounding boxes, sized [4].
    :param box2: bounding boxes, sized [4].
    :return:
    """

    if (box1[2] < box2[0] or box2[2] < box1[0] or box1[3] < box2[1] or box2[3] < box1[1]):
        return 0.0
    
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box_true_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box_pred_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = inter_area / float(box_true_area + box_pred_area - inter_area)
    return iou


def box_nms(bboxes, scores, threshold=0.5, mode='min'):
    """Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    # print(bboxes.shape)
    if len(bboxes.shape) > 1:
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
    else:
        return None

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:

        if order.numel() == 1:
            break
            
        i = order[0]
        keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


def softmax(x):
    '''Softmax along a specific dimension.

    Args:
      x: (tensor) input tensor, sized [N,D].

    Returns:
      (tensor) softmaxed tensor, sized [N,D].
    '''
    xmax, _ = x.max(1)
    x_shift = x - xmax.view(-1,1)
    x_exp = x_shift.exp()
    return x_exp / x_exp.sum(1).view(-1,1)


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    
    # print('ohe.labels  ', labels, labels.shape)
    # print('ohe.y[labels]  ', y[labels], y[labels].shape)
    return y[labels]            # [N,D]


def msr_init(net):
    '''Initialize layer parameters.'''
    for layer in net:
        if type(layer) == nn.Conv2d:
            n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2./n))
            layer.bias.data.zero_()
        elif type(layer) == nn.BatchNorm2d:
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif type(layer) == nn.Linear:
            layer.bias.data.zero_()


def get_true_detection(labels_path: str) -> pd.DataFrame:
    """ Function makes dataframe with true labels for detection task

    :param labels_path: path to the detection labels
    :return:
    """
    if os.path.exists(labels_path):
        true_df = pd.DataFrame()
        true_names, true_boxes, true_labels = [], [], []
        with open(labels_path) as f:
            lines = f.readlines()
        for line in lines:
            splited = line.strip().split()
            true_names.append(splited[0])
            num_boxes = (len(splited) - 1) // 5

            box = [
                [int(splited[1 + 5 * i]), int(splited[2 + 5 * i]), int(splited[3 + 5 * i]),
                 int(splited[4 + 5 * i])]
                for i in range(num_boxes)
            ]
            label = [int(splited[5 + 5 * i]) for i in range(num_boxes)]
            true_boxes.append(box)
            true_labels.append(label)
        true_df['names'] = true_names
        true_df['boxes'] = true_boxes
        true_df['labels'] = true_labels
    else:
        raise ValueError(
            f"Labels path {labels_path} does not exists."
        )
    return true_df


def get_true_classification(labels_path: str) -> pd.DataFrame:
    """ Function makes dataframe with true labels for classification task

    :param labels_path: path to the classification labels
    :return:
    """
    if os.path.exists(labels_path):
        true_df = pd.read_csv(filepath_or_buffer=os.path.join(labels_path), sep=';')
    else:
        raise ValueError(
            f"Labels path {labels_path} does not exists."
        )
    return true_df


def get_true_segmentation(labels_path: str, mode: str, size: (tuple, int)) -> pd.DataFrame:
    """ Function makes dataframe with true labels for segmentation task

    :param labels_path: path to the classification labels
    :param mode: mode (binary or multi)
    :param colors: colors encoding (for multi only)
    :param n_classes: numbers of classes
    :return:
    """
    if os.path.exists(labels_path):
        true_df = pd.DataFrame()
        if mode == 'binary':
            true_masks = []
            mask_names = os.listdir(labels_path)
            for mask_name in mask_names:
                mask = cv2.imread(filename=os.path.join(labels_path, mask_name), flags=0)
                mask = mask / 255.
                true_masks.append(mask)
            true_df['names'] = mask_names
            true_df['masks'] = true_masks

        # Get true labels for multiclass segmentation
        elif mode == 'multi':  # self.mode == 'multi'
            true_masks = []
            mask_names = os.listdir(labels_path)
            for mask_name in mask_names:
                mask = cv2.imread(filename=os.path.join(labels_path, mask_name))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                # ToDo: remove hardcode
                from .image_utils import pad, resize
                # mask = resize_image(mask, size=size, interpolation=cv2.INTER_NEAREST)
                mask = pad(mask)
                mask = resize(mask, size=size, interpolation=cv2.INTER_NEAREST)
                # from .image_utils import draw_images
                # draw_images([mask])

                true_masks.append(mask)
            true_df['names'] = mask_names
            true_df['masks'] = true_masks
        else:
            raise ValueError(
                f"Wrong mode parameter: {mode}. Should be 'binary' or 'multi'."
            )
    else:
        raise ValueError(
            f"Labels path {labels_path} does not exists."
        )
    return true_df
