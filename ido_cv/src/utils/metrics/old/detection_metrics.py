# -*- coding: utf-8 -*-
"""
Module implements metrics for detection:
    - mean average precision

"""
from copy import deepcopy
import numpy as np
from ...utils import common_utils


def mean_ap(true_df, pred_df, iou_thresholds):
    """ Function to evaluate mean Average Precision score of the model

    :param true_df: Dataframe with true images and labels
    :param pred_df: Dataframe with predicted images and labels
    :param iou_thresholds:
    :return:
    """
    all_df = true_df.merge(pred_df, on='names', suffixes=['_true', '_pred'])

    # Get dictionaries for the mAP evaluator
    classes_true = dict()
    classes_pred = dict()
    for cls in range(0, 10):
        class_true = dict()
        class_pred = dict()
        for row in all_df.iterrows():
            name = row[1]['names']

            # Get dict of gt bboxes for each class
            boxes_t = row[1]['boxes_true']
            labels_t = row[1]['labels_true']
            true_boxes = [bbox for bbox, label in zip(boxes_t, labels_t) if cls == label]
            class_true[name] = true_boxes
            pred_dict = dict()
            boxes_p = row[1]['boxes_pred']
            labels_p = row[1]['labels_pred']
            scores_p = row[1]['scores']
            pred_boxes = [bbox.tolist() for bbox, label in zip(boxes_p, labels_p) if cls == label]
            pred_scores = [score for score, label in zip(scores_p, labels_p) if cls == label]
            pred_dict['boxes'] = pred_boxes
            pred_dict['scores'] = pred_scores
            class_pred[name] = pred_dict
        classes_true[cls] = class_true
        classes_pred[cls] = class_pred

    # Get scores
    if isinstance(iou_thresholds, float):
        iou_thresholds = [iou_thresholds]
    elif isinstance(iou_thresholds, tuple) and len(iou_thresholds) == 3:
        first = iou_thresholds[0]
        last = iou_thresholds[1]
        n = iou_thresholds[2]
        iou_thresholds = np.linspace(first, last, n)
    else:
        raise ValueError(f'Wrong iou_thresholds value: {iou_thresholds}')
    ap_scores = []
    for cls in range(0, 10):
        true_boxes = classes_true[cls]
        pred_boxes = classes_pred[cls]
        cls_avg_precs = []
        cls_iou_thrs = []
        for iou_thr in iou_thresholds:
            data = get_avg_precision_at_iou(true_boxes, pred_boxes, iou_thr=iou_thr)
            cls_avg_precs.append(data['avg_prec'])
            cls_iou_thrs.append(iou_thr)
        avg_precs = [float('{:.4f}'.format(ap)) for ap in cls_avg_precs]
        iou_thrs = [float('{:.4f}'.format(thr)) for thr in cls_iou_thrs]
        print('mAP for class {}: {:.4f}'.format(cls, np.mean(cls_avg_precs)))
        print('avg precs: ', avg_precs)
        print('iou_thrs:  ', iou_thrs)
        ap_scores.append(np.mean(cls_avg_precs))
    return ap_scores


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = common_utils.box_iou_alt(gt_box, pred_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)


def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map


def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: avg precision as well as summary info about the PR curve
        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(
                gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs
    }
