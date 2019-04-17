# -*- coding: utf-8 -*-
"""
    Модлль постпроцессинга
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import remove_small_holes, remove_small_objects

from .. import allowed_parameters

COLORS = allowed_parameters.SEG_MULTI_COLORS


def delete_small_instances(image, hole_size=None, obj_size=None):
    """ Function delete small objects and holws on the given image
    
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
            # image = remove_small_objects(image, min_size=obj_size)
            out_image = remove_obj(image_, min_size=obj_size)
    else:
        raise ValueError(
            f'Wrong image shape: {image.shape}'
        )
    return out_image


def remove_obj(image, min_size=10):
    image_ = np.copy(image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image_, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2
    
    
def get_dataframes(path_list):
    """ Function to make list of dataframes
    
    :param path_list:
    :return:
    """
    df_list = []
    for path in path_list:
        
        df = pd.DataFrame()
        names = os.listdir(path)
        masks = []
        for name in names:
            mask = cv2.imread(os.path.join(path, name), 0)
            masks.append(mask)
            
        df['names'] = names
        df['masks'] = masks
        df_list.append(df)
    return df_list
    
    
def get_stacked_dataframe(df_list):
    """ Function make stack
    
    :param df_list:
    :return:
    """
    all_df = df_list[0]
    for i, df in enumerate(df_list[1:]):
        all_df = all_df.merge(df, on='names', suffixes=[str(i - 1), str(i)])
    stacked_masks = []
    for row in all_df.iterrows():
        masks = np.asarray([row[1].iloc[i] for i in range(1, row[1].shape[0])])
        stacked_masks.append(masks)
    stacked_df = pd.DataFrame()
    stacked_df['names'] = all_df['names']
    stacked_df['masks'] = stacked_masks
    return stacked_df


def combine_bin_mult(stacked_df):
    """ Function returns combined mask for binary and multiclass segmentations
    
    :param stacked_df:
    :return:
    """
    out_df = pd.DataFrame()
    masks = []
    for row in stacked_df.iterrows():
        stacked_masks = row[1]['masks']
        mask_b = stacked_masks[0, :, :]
        mask_m = stacked_masks[1, :, :]
        mask_b[mask_b > 0] = mask_m[mask_b > 0]
        mask_b = make_erode(mask_b.astype(np.uint8), kernel_size=(2, 1))
        masks.append(mask_b)
    out_df['names'] = stacked_df['names']
    out_df['masks'] = masks
    return out_df


def get_dates(pred_df: pd.DataFrame):
    dates = []
    for row in pred_df.iterrows():
        mask = row[1]['masks']
        date = get_date_from_image(image=mask, classes=11)
        dates.append(date)
    pred_df['dates'] = dates
    return pred_df


def get_date_from_image(image, classes):
    """ Function return a suggested date from the given data

    :param pred_df:
    :return:
    """
    
    if len(image.shape) == 2:
        all_letters = []
        for clr in range(classes):
            if clr == 0:
                continue

            number = clr - 1
            img = np.zeros_like(image, dtype=np.uint8)
            img[image == clr] = 1
            
            if np.sum(img) == 0:
                continue
                
            im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None:
                hier = [np.squeeze(hierarchy[:, i, :], axis=0) for i in range(hierarchy.shape[1])]
                cnt_list = []
                for c, h in zip(contours, hier):
                    if h[-1] == -1:
                        cnt_list.append(c)
    
                rects = [cv2.boundingRect(c) for c in cnt_list]
                rects = [(r[0], r[1], r[0] + r[2], r[1] + r[3]) for r in rects]
                im2 = cv2.drawContours(im2, contours, -1, 155, 1)
                letter = []
                for r in rects:
                    if (r[2] - r[0]) > 2 and (r[3] - r[1]) > 2:
                        letter.append((number, r))
                        im2 = cv2.rectangle(im2, (r[0], r[1]), (r[2], r[3]), 155, 1)
                all_letters += letter
        all_letters = [i[0] for i in sorted(all_letters, key=lambda x: x[1][0])]
        return all_letters
    else:
        raise ValueError(
            f'Wrong shape of image: {image.shape}. Should be 2-d image with values from 0 to '
            f'classes.'
        )


def make_erode(image, kernel_size=(100, 1)):
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
        

def _draw_images(images_list):
    n_images = len(images_list)

    fig = plt.figure()
    for i, image in enumerate(images_list):
        ax = fig.add_subplot(n_images, 1, i + 1)
        ax.imshow(image)
    plt.show()
