# -*- coding: utf-8 -*-
"""
    Perform image transformations
"""
import math
import random
import cv2
import numpy as np
import torch
from skimage import util


def pad(img, boxes=None, mode='constant'):
    """ Function for padding image and (optionally) bboxes

    :param img: Image
    :param boxes: Bboxes
    :return:
    """
    # TODO make pad universal
    img_largest_size = img.shape[1]
    img_shortest_size = img.shape[0]
    top_add = int(np.floor((img_largest_size-img_shortest_size)/2))
    bot_add = int(np.ceil((img_largest_size-img_shortest_size)/2))
    pad_width = ((top_add, bot_add), (0, 0))
    
    if len(img.shape) == 3:
        out_image = np.ndarray(shape=(img_largest_size, img_largest_size, img.shape[2]),
                               dtype=np.uint8)
        for ch in range(out_image.shape[2]):
            img_to_pad = img[:, :, ch]
            out_image[:, :, ch] = util.pad(img_to_pad, pad_width, mode=mode)
    elif len(img.shape) == 2:
        out_image = util.pad(img, pad_width, mode=mode)
    else:
        raise ValueError(f'Wrong shape of image: {img.shape}')
    
    if boxes is not None:
        boxes = boxes + torch.Tensor([0, top_add, 0, top_add])
        return out_image, boxes
    return out_image


def unpad(img, boxes=None, img_shape=(45, 256)):
    # TODO make unpad universal
    img_largest_size = img_shape[1]
    img_shortest_size = img_shape[0]
    top_add = int(np.floor((img_largest_size - img_shortest_size) / 2))
    # bot_add = int(np.ceil((img_largest_size - img_shortest_size) / 2))
    
    if len(img.shape) == 3:
        out_image = np.ndarray(shape=(img_shape[0], img_shape[1], img.shape[2]),
                               dtype=np.float32)
        for ch in range(out_image.shape[2]):
            out_image[:, :, ch] = img[top_add:top_add + img_shortest_size, :, ch]
    elif len(img.shape) == 2:
        out_image = np.ndarray(shape=(img_shape[0], img_shape[1]), dtype=np.float32)
        out_image[:, :] = img[top_add:top_add + img_shortest_size, :]
    else:
        raise ValueError(f'Wrong shape of image: {img.shape}')

    if boxes is not None:
        out_boxes = boxes - np.asarray([0, top_add, 0, top_add])
        out_boxes = np.maximum(out_boxes, 0).astype(np.uint64)
        return out_image, out_boxes
    return out_image


def resize(img, boxes=None, size=256, max_size=10000, interpolation=cv2.INTER_AREA):
    """ Resize the input PIL image to the given size.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    """
    w, h = img.shape[0], img.shape[1]
    if isinstance(size, int):
        size_min = min(w, h)
        size_max = max(w, h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h

    out_image = cv2.resize(img, (ow, oh), interpolation=interpolation)
    if boxes is not None:
        out_boxes = boxes * torch.Tensor([sw, sh, sw, sh])
        return out_image, out_boxes
    return out_image


def resize_image(image: np.ndarray, size: tuple, interpolation=cv2.INTER_AREA):
    """ Function resize given image to the size

    :param image: input image
    :param size: Size to resize
    :param interpolation: Interpolation method
    :return:
    """
    h_i, w_i = image.shape[0], image.shape[1]
    h, w = size
    if h == h_i and w == w_i:
        return image
    image = cv2.resize(image, (w, h), interpolation=interpolation)
    return image


def resize_bboxes(boxes, size_from=256, size_to=281):
    """ Function to resize bboxes

    :param boxes: List of bboxes
    :param size_from: Size of image resize from
    :param size_to: Size of image resize to
    :return: Resized bboxes
    """
    ow, oh = size_to, size_to
    sw = float(ow) / size_from
    sh = float(oh) / size_from
    out_boxes = boxes * np.asarray([sw, sh, sw, sh])
    out_boxes = out_boxes.astype(np.uint64)
    return out_boxes


def unpad_bboxes(boxes, img_shape):
    """ Make unpaded bboxes

    :param boxes: List of bboxes
    :param img_shape: Shape of image
    :return: Unpaded bboxes
    """
    img_largest_size = img_shape[1]
    img_shortest_size = img_shape[0]
    top_add = int(np.floor((img_largest_size - img_shortest_size) / 2))
    out_boxes = boxes - np.asarray([0, top_add, 0, top_add])
    out_boxes = np.maximum(out_boxes, 0).astype(np.uint64)
    return out_boxes


def random_crop(img, boxes):
    """ Crop the given PIL image to a random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    """
    success = False
    for _ in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x + w, y + h))
    boxes -= torch.Tensor([x, y, x, y])
    boxes[:, 0::2].clamp_(min=0, max=w - 1)
    boxes[:, 1::2].clamp_(min=0, max=h - 1)
    return img, boxes


def center_crop(img, boxes, size):
    """ Crops the given PIL Image at the center.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size (tuple): desired output size of (w,h).

    Returns:
      img: (PIL.Image) center cropped image.
      box
    """
    w, h = img.size
    ow, oh = size
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j + ow, i + oh))
    boxes -= torch.Tensor([j, i, j, i])
    boxes[:, 0::2].clamp_(min=0, max=ow - 1)
    boxes[:, 1::2].clamp_(min=0, max=oh - 1)
    return img, boxes


def make_colors(image):
    image[(image < 240) & (image > 50)] = 0
    return image