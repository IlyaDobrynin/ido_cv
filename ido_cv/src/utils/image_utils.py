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
from skimage.morphology import remove_small_holes, remove_small_objects
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15.0, 12.0)


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


def draw_images(images_list: list, orient: str = 'horizontal'):
    """ Function draws images from images_list

    :param images_list: List of numpy.ndarray images
    :param orient: How to orient images ('horisontal' or 'vertical')
    :return:
    """
    n_images = len(images_list)
    fig = plt.figure()
    for i, image in enumerate(images_list):
        if orient == 'horizontal':
            ax = fig.add_subplot(1, n_images, i + 1)
        elif orient == 'vertical':
            ax = fig.add_subplot(n_images, 1, i + 1)
        else:
            raise ValueError(
                f"Wrong parameter orient: {orient}. Should be 'horizontal' or 'vertical'."
            )
        ax.imshow(image)
    plt.show()


def convert_multilabel_mask(mask: np.ndarray, colors: dict, how: str = 'rgb2class',
                            ignore_class: int = None) -> np.ndarray:
    """ Function for multilabel mask convertation

    :param mask: Numpy ndarray of mask
    :param colors: Dictionary with colors encodings
    :param how: 'rgb2class' or 'class2rgb'
    :param ignore_class: Class to ignore
    :return:
    """
    n_classes = len(colors.keys())
    if how == 'rgb2class':
        out_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for cls in range(n_classes):
            if cls == ignore_class:
                continue
            matching = np.all(mask == colors[cls], axis=-1)
            out_mask[matching] = cls + 1
    elif how == 'class2rgb':
        out_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls in range(n_classes):
            if cls == ignore_class:
                continue
            matching = (mask[:, :] == cls)
            out_mask[matching, :] = colors[cls - 1]
    else:
        raise ValueError(
            f"Wrong parameter how: {how}. Should be 'rgb2class or class2rgb."
        )
    return out_mask


def delete_small_instances(image: np.ndarray, hole_size: int = None, obj_size: int = None):
    """ Function delete small objects and holes on the given image

    :param image: Input image
    :param hole_size: Maximum area of the hole that will be filled
    :param obj_size: Minimum size of the object to delete
    :return:
    """
    image_ = np.copy(image)
    if len(image.shape) == 3:
        out_image = np.zeros_like(image_, dtype=np.uint8)
        for ch in range(image_.shape[-1]):
            img = image_[:, :, ch]
            if hole_size is not None:
                img = remove_small_holes(img, area_threshold=hole_size)
            if obj_size is not None:
                img = remove_small_objects(img, min_size=obj_size)
            out_image[:, :, ch] = img

    elif len(image.shape) == 2:
        out_image = np.copy(image_)
        if hole_size is not None:
            out_image = remove_small_holes(image_, area_threshold=hole_size)
        if obj_size is not None:
            out_image = remove_obj(image_, min_size=obj_size)
    else:
        raise ValueError(
            f'Wrong image shape: {image.shape}'
        )
    return out_image


def remove_obj(image: np.ndarray, min_size: int = 10, connectivity: int = 8):
    image_ = np.copy(image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image_,
        connectivity=connectivity
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


def make_erode(image: np.ndarray, kernel_size: tuple = (100, 1)):
    """ Function make morphological erosion for a given image

    :param image:
    :param kernel_size:
    :return:
    """
    kernel = np.ones(kernel_size, np.uint8)
    erosion = cv2.erode(image, kernel).astype(np.uint8)
    return erosion


def make_dilate(image: np.ndarray, kernel_size=(100, 1)):
    """ Function make morphological dilation for a given image

    :param image:
    :param kernel_size:
    :return:
    """
    kernel = np.ones(kernel_size, np.uint8)
    dilation = cv2.dilate(image, kernel).astype(np.uint8)
    return dilation
