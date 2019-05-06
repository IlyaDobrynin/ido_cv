import numpy as np
from tqdm import tqdm
import cv2
from .image_utils import resize_image


def get_opt_threshold(trues, preds, threshold_min=0.1, metric_name='dice', device='cpu'):
    """ Function returns optimal threshold for prediction images

    :param trues: Ground truth validation masks
    :param preds: Predicted validation masks
    :param threshold_min: Minimum threshold value
    :param metric_name: Metric for threshold calculation
    :param device: Device to calculate metric
    :return:
    """
    thresholds = np.linspace(0, 1, 50)
    preds_new = [np.uint8(preds > threshold) for threshold in tqdm(thresholds)]

    if device == 'cpu':
        metrics = np.array([numpy_metric(trues, pred, metric_name=metric_name)
                            for pred in tqdm(preds_new)])
    elif device == 'gpu':
        metrics = np.array([torch_metric(trues, pred, metric_name=metric_name)
                            for pred in tqdm(preds_new)])
    else:
        raise ValueError(
            f'Wrong device parameter: {device}. Should be "cpu" or "gpu".'
        )
    threshold_best_index = np.argmax(metrics)
    best_metric = metrics[threshold_best_index]
    best_threshold = thresholds[threshold_best_index] \
        if thresholds[threshold_best_index] > threshold_min else threshold_min
    return best_threshold, best_metric


def torch_metric(trues, preds, metric_name):
    """ Function returns metrics calculated on GPU via pytorch

    :param preds: Predictions of the network
    :param trues: Ground truth labels
    :param metric_name: Name of the metric to calculate
    """
    smooth = 1e-12
    metrics = []

    for true, pred in zip(trues, preds):
        true = (true == 1)
        pred = (pred == 1)

        if true.long().sum() == 0 and pred.long().sum() > 0:
            metrics.append(0)
            continue
        if true.long().sum() > 0 and pred.long().sum() == 0:
            metrics.append(0)
            continue
        if true.long().sum() == 0 and pred.long().sum() == 0:
            continue

        union = (true | pred).long().sum().float()
        intersection = (true & pred).long().sum().float()

        if metric_name == 'dice':
            metric = (2.0 * intersection + smooth) / (union + intersection + smooth)
            metric = metric.data.cpu().numpy()
        elif metric_name == 'jaccard':
            metric = (intersection + smooth) / (union + smooth)
            metric = metric.data.cpu().numpy()
        elif metric_name == 'm_iou':
            iou = (intersection + smooth) / (union + smooth)
            iou = iou.data.cpu().numpy()
            thresholds = np.arange(0.5, 1, 0.05)
            s = []
            for thresh in thresholds:
                s.append(iou > thresh)
            metric = np.mean(s)
        else:
            raise ValueError(
                f'Wrong metric_name: {metric_name}. Should be "dice", "jaccard" or "m_iou".'
            )
        metrics.append(metric)
    return np.mean(metrics)


def numpy_metric(trues, preds, metric_name):
    """ Function returns metric (dice, jaccard or mean IoU)

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param metric_name:
    :param device: Device to calculate metric:
                    - 'cpu'
                    - 'gpu'
    :return:
    """
    smooth = 1e-12
    metrics = []
    for true, pred in zip(trues, preds):
        if np.count_nonzero(true) == 0 and np.count_nonzero(pred) > 0:
            metrics.append(0)
            continue
        if np.count_nonzero(true) > 0 and np.count_nonzero(pred) == 0:
            metrics.append(0)
            continue
        if np.count_nonzero(true) == 0 and np.count_nonzero(pred) == 0:
            continue

        pred = resize_image(pred, size=true.shape[:2], interpolation=cv2.INTER_NEAREST)
        true_bool = np.asarray(true, dtype=bool)
        pred_bool = np.asarray(pred, dtype=bool)
        intersection = np.sum(np.logical_and(true, pred).astype(np.uint8))
        union = true_bool.sum() + pred_bool.sum()

        if metric_name == 'dice':
            metric = (2.0 * intersection + smooth) / (union + smooth)
        elif metric_name == 'jaccard':
            metric = (intersection + smooth) / (union - intersection + smooth)
        elif metric_name == 'm_iou':
            iou = (intersection + smooth) / (union - intersection + smooth)
            thresholds = np.arange(0.5, 1, 0.05)
            s = []
            for thresh in thresholds:
                s.append(iou > thresh)
            metric = np.mean(s)
        else:
            raise ValueError(
                f'Wrong metric_name: {metric_name}. Should be "dice", "jaccard" or "m_iou".'
            )
        metrics.append(metric)
    return np.mean(metrics)


def calculate_confusion_matrix_from_arrays(prediction: np.ndarray, ground_truth: np.ndarray,
                                           nr_labels: int) -> np.ndarray:
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def get_metric_from_matrix(confusion_matrix: np.ndarray, metric_name: str,
                           ignore_class: int = None) -> list:
    metrics = []
    if ignore_class is None:
        index_list = [i for i in range(confusion_matrix.shape[0])]
    else:
        index_list = [i for i in range(confusion_matrix.shape[0]) if i != ignore_class]

    for index in index_list:

        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives

        if metric_name == 'jaccard':
            denom = true_positives + false_positives + false_negatives
        elif metric_name == 'dice':
            denom = 2 * true_positives + false_positives + false_negatives
        else:
            raise ValueError(
                f'Wrong metric name: {metric_name}. Should be dice or jaccard.'
            )
        if denom == 0:
            iou = 0
        else:
            if metric_name == 'jaccard':
                iou = float(true_positives) / denom
            else:
                iou = 2 * float(true_positives) / denom
        metrics.append(iou)
    return metrics