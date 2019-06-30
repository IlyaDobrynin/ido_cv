# -*- coding: utf-8 -*-
"""
    Encode object boxes and labels.
"""

import numpy as np
import torch
from ..common_utils import box_iou, box_nms


class DataEncoder:
    def __init__(self, input_size):
        self.image_shape = np.asarray([input_size, input_size, 3])
        self.pyramid_levels = np.asarray([3, 4, 5], dtype=np.uint16)
        self.base_strides = np.asarray([8, 16, 32], dtype=np.uint16)
        self.anchor_sizes = np.asarray([16, 32, 64], dtype=np.float32)
        self.stride_step = 1
        self.anchor_strides = self.base_strides * self.stride_step
        # print('dataEncoder.anchor_strides', self.anchor_strides)
        self.anchor_ratios = [1/3., 1/1., 3/1.]
        self.anchor_scales = [1., pow(2, 1/3.), pow(2, 2/3.)]

        self.fm_sizes = self._get_feature_maps_sizes()
        self.anchor_wh = self._get_anchors()
        self.num_anchors = int(len(self.anchor_ratios) * len(self.anchor_scales))

    def _get_anchors(self):
        """ Function returns tensor of base anchors with size (N, N_anchors, 4)
        :return:
        """
        all_anchors = np.zeros((0, 4))
        for idx, size in enumerate(self.anchor_sizes):
            anchors = self._get_anchors_for_size(
                base_size=size,
                ratios=self.anchor_ratios,
                scales=self.anchor_scales
            )
            # print("encoder.anchors", anchors)
            shifted_anchors = self.shift(self.fm_sizes[idx], self.anchor_strides[idx], anchors)
            # print("encoder.shifted_anchors", shifted_anchors.shape, shifted_anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        out_tensor = torch.Tensor(all_anchors)
        return out_tensor

    def _get_anchors_for_size(self, base_size=16., ratios=None, scales=None):
        """ Generate anchor (reference) windows by enumerating aspect ratios X
            scales w.r.t. a reference window.
        """
        num_anchors = len(ratios) * len(scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        # print(anchors)
        return anchors

    def _get_feature_maps_sizes(self):
        """ Function returns size of feature maps for different pyramid levels

        :return:
        """
        image_shape = np.array(self.image_shape[:2])
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        # print("encoder._get_fm_sizes", image_shapes)
        return image_shapes

    def shift(self, shape, stride, anchors):
        """ Produce shifted anchors based on shape of the map and stride size.

        Args
            shape  : Shape to shift the anchors over.
            stride : Stride to shift the anchors with over the shape.
            anchors: The anchors to apply at each location.
        """

        # create a grid starting from half stride from the top left corner
        shift_x = (np.arange(0, shape[1] * int(1/self.stride_step)) + 0.5) * stride
        shift_y = (np.arange(0, shape[0] * int(1/self.stride_step)) + 0.5) * stride
        
        # print('encoder.shift_x:', shift_x.shape, shift_x)
        # print('encoder.shift_y:', shift_y.shape, shift_y)

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()
        # print(shifts.shape)
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        return all_anchors

    def encode(self, boxes, labels):
        """ Function encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx_1 = (x - anchor_x) / anchor_w
          ty_1 = (y - anchor_y) / anchor_h
          tx_2 = (x - anchor_x) / anchor_w
          ty_2 = (y - anchor_y) / anchor_h

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].+
        """
        # Get base anchors
        anchor_boxes = self._get_anchors()
        
        # Find IoU for base anchors and label boxes
        ious = box_iou(anchor_boxes, boxes, order='xyxy')

        # Find max IoU and its index
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        # Make coding process
        anchor_widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
        anchor_heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]

        targets_dx1 = (boxes[:, 0] - anchor_boxes[:, 0]) / anchor_widths
        targets_dy1 = (boxes[:, 1] - anchor_boxes[:, 1]) / anchor_heights
        targets_dx2 = (boxes[:, 2] - anchor_boxes[:, 2]) / anchor_widths
        targets_dy2 = (boxes[:, 3] - anchor_boxes[:, 3]) / anchor_heights

        targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
        targets = targets.T

        # import matplotlib.pyplot as plt
        # import cv2
        # anchor_boxes_np = anchor_boxes.data.cpu().numpy()
        # image = np.zeros(shape=self.image_shape)
        # draw = image.copy()
        # print("-" * 30, "\n")
        # for box in anchor_boxes_np[3:1203:24, :]:
        #     # lt = (int(np.maximum(0, box[0])), int(np.maximum(0, box[1])))
        #     # rb = (int(np.minimum(256, box[2])), int(np.minimum(256, box[3])))
        #     lt = (box[0], box[1])
        #     rb = (box[2], box[3])
        #     print(lt, rb)
        #     draw = cv2.rectangle(image, lt, rb, (0, 255, 0), 1)
        # plt.imshow(draw)
        # plt.show()

        loc_targets = torch.from_numpy(targets)
        cls_targets = 1 + labels[max_ids]

        # # if max iou < 0.4, assume that there is background on the box
        cls_targets[max_ious < 0.5] = 0
        # # if 0.4 < max iou < 0.5, assume that this box is not certain and ignore it
        cls_targets[(max_ious > 0.4) & (max_ious < 0.5)] = -1

        # Simple code for checking samples

        # print("encoder.labels:", labels)
        # num_dict = {}
        # for i in range(-1, 11):
        #     n_entries = cls_targets[cls_targets == i].shape
        #     num_dict[i] = n_entries
        # for k, v in num_dict.items():
        #     print(f"Число {k-1}: {v}")
        return loc_targets.float(), cls_targets

    def decode(self, loc_preds, cls_preds, cls_thresh=0.3, nms_thresh=0.7):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        anchor_boxes = self._get_anchors()
        
        anchor_widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
        anchor_heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]

        # Decode boxes
        targets_dx1 = loc_preds[:, 0] * anchor_widths  + anchor_boxes[:, 0]
        targets_dy1 = loc_preds[:, 1] * anchor_heights + anchor_boxes[:, 1]
        targets_dx2 = loc_preds[:, 2] * anchor_widths  + anchor_boxes[:, 2]
        targets_dy2 = loc_preds[:, 3] * anchor_heights + anchor_boxes[:, 3]
        boxes = torch.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2), dim=1)

        score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        # print('encoder.score:', score, score.shape)
        # print('encoder.labels:', labels)
        ids = score > cls_thresh
        # print(ids)
        ids = ids.nonzero().squeeze()             # [#obj,]
        # print(ids.shape)
        keep = box_nms(boxes[ids], score[ids], threshold=nms_thresh)
        # print(keep)
        if keep is not None:
            # print(boxes[ids])
            out_boxes = boxes[ids][keep]
            out_labels = labels[ids][keep]
            out_score = score[ids][keep]
            # print('decoder.score', score[ids], score[ids].shape)
            return (out_boxes, out_labels, out_score)
        else:
            return None
