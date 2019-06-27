# -*- coding: utf-8 -*-
"""
Module implements functions for work with dataset

"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset
from mts_cv.src.utils.image_utils import draw_images
from mts_cv.src.utils.image_utils import convert_multilabel_mask


def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[1, row, :] = const
    image_tensor[2] = image_tensor[0] * image_tensor[1]
    return image_tensor


class BinSegDataset(Dataset):
    """ Class describes current dataset
    """
    
    def __init__(
            self,
            train:          bool,
            add_depth:      bool = False,
            data_path:      str = None,
            data_file:      pd.DataFrame = None,
            common_augs     =None,
            train_time_augs =None,
            show_sample:    bool = False
    ):

        self.train = train
        self.show_sample = show_sample
        if data_path is not None:
            self.data_path = data_path
            self.file_names = os.listdir(os.path.join(self.data_path, 'images'))
            self.from_path = True
        elif data_file is not None:
            self.file_names = data_file['names'].values.tolist()
            self.data_file = data_file
            self.from_path = False
        else:
            raise ValueError(
                f"data_path or data_file should be provided"
            )

        self.common_augs = common_augs
        self.train_time_augs = train_time_augs
        self.add_depth = add_depth

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        fname = self.file_names[idx]
        if self.from_path:
            image = cv2.imread(os.path.join(self.data_path, r'images/{}'.format(fname)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = self.data_file[self.data_file['names'] == fname]['images'].values[0]
        if self.train:
            if self.from_path:
                mask = cv2.imread(os.path.join(self.data_path, r'masks/{}'.format(fname)), 0)
            else:
                mask = self.data_file[self.data_file['names'] == fname]['masks'].values[0]
            return self._get_trainset(image, mask, fname)
        else:
            return self._get_testset(image, fname)
    
    def _get_trainset(self, image, mask, name):
        if self.common_augs:
            data = {'image': image, 'mask': mask}
            common_augs = self.common_augs(**data)
            image, mask = common_augs['image'], common_augs['mask']
        if self.train_time_augs:
            data = {'image': image, 'mask': mask}
            augmented = self.train_time_augs(**data)
            image, mask = augmented['image'], augmented['mask']
        if self.add_depth:
            image = add_depth_channels(img_to_tensor(image))

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        image = img_to_tensor(image)
        mask = img_to_tensor(mask)

        if self.show_sample:
            viz_image = np.moveaxis(image.data.numpy(), 0, -1)
            viz_mask = np.squeeze(np.moveaxis(mask.data.numpy(), 0, -1), -1)
            msk_img = np.copy(viz_image)
            matching = np.all(np.expand_dims(viz_mask, axis=-1) > 0.1, axis=-1)
            msk_img[matching, :] = [0, 0, 0]
            draw_images([viz_image, viz_mask, msk_img])

        return image, mask, str(name)
    
    def _get_testset(self, image, name):
        if self.common_augs:
            data = {'image': image}
            common_augs = self.common_augs(**data)
            image = common_augs['image']

        if self.add_depth:
            image = add_depth_channels(img_to_tensor(image))

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image = img_to_tensor(image)
        if self.show_sample:
            viz_image = np.moveaxis(image.data.numpy(), 0, -1)
            draw_images([viz_image])

        return image, image.shape, str(name)

    def collate_fn(self, batch):
        '''Pad images and encode targets.
    
        As for images are of different sizes, we need to pad them to the same size.
    
        Args:
          batch: (list) of images, cls_targets, loc_targets.
    
        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        # ToDo add checks
        h, w = imgs[0].shape[1], imgs[0].shape[2]
        if self.train:
            masks = [x[1] for x in batch]
            names = [x[2] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            targets = torch.zeros(num_imgs, 1, h, w)
            for i in range(num_imgs):
                inputs[i] = imgs[i]
                targets[i] = masks[i]
            return inputs, targets, names
        else:
            names = [x[-1] for x in batch]
            shapes = [x[1] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            for i in range(num_imgs):
                inputs[i] = imgs[i]
            return inputs, shapes, names


class MultSegDataset(Dataset):
    """ Class describes current dataset
    """
    
    def __init__(
            self,
            train: bool,
            add_depth: bool = False,
            data_path: str = None,
            data_file: pd.DataFrame = None,
            common_augs=None,
            train_time_augs=None,
            show_sample: bool = False,
            **kwargs
    ):

        self.train = train
        self.show_sample = show_sample
        if data_path is not None:
            self.data_path = data_path
            self.file_names = os.listdir(os.path.join(self.data_path, 'images'))
            self.from_path = True
        elif data_file is not None:
            self.file_names = data_file['names'].values.tolist()
            self.data_file = data_file
            self.from_path = False

        self.colors = kwargs['label_colors']
        self.common_augs = common_augs
        self.train_time_augs = train_time_augs
        self.add_depth = add_depth
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx: int):
        fname = self.file_names[idx]
        if self.from_path:
            image = cv2.imread(os.path.join(self.data_path, fr'images/{fname}'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = self.data_file[self.data_file['names'] == fname]['images'].values[0]
        if self.train:
            if self.from_path:
                mask = cv2.imread(os.path.join(self.data_path, f'masks/{fname}'))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                mask = self.data_file[self.data_file['names'] == fname]['masks'].values[0]
            return self._get_trainset(image, mask, fname)
        else:
            return self._get_testset(image, fname)
    
    def _get_trainset(self, image: np.ndarray, mask: np.ndarray, name: str):
        mask = convert_multilabel_mask(mask, label_colors=self.colors, how='rgb2class')
        if self.common_augs:
            data = {'image': image, 'mask': mask}
            common_augs = self.common_augs(**data)
            image, mask = common_augs['image'], common_augs['mask']
        if self.train_time_augs:
            data = {'image': image, 'mask': mask}
            augmented = self.train_time_augs(**data)
            image, mask = augmented['image'], augmented['mask']
        if self.add_depth:
            image = add_depth_channels(img_to_tensor(image))

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            
        image = img_to_tensor(image)
        mask = torch.from_numpy(np.moveaxis(mask, -1, 0)).long()

        if self.show_sample:
            viz_image = np.moveaxis(image.data.numpy(), 0, -1)
            viz_mask = np.moveaxis(mask.data.numpy(), 0, -1)[:, :, 0]
            draw_images([viz_image, viz_mask])
        
        return image, mask, str(name)
    
    def _get_testset(self, image, name):
        if self.common_augs:
            data = {'image': image}
            common_augs = self.common_augs(**data)
            image = common_augs['image']
        if self.add_depth:
            image = add_depth_channels(img_to_tensor(image))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = img_to_tensor(image)
        return image, str(name)

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        # ToDo add checks
        h, w = imgs[0].shape[1], imgs[0].shape[2]
        if self.train:
            masks = [x[1] for x in batch]
            names = [x[2] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            targets = torch.zeros(num_imgs, 1, h, w).long()
            for i in range(num_imgs):
                inputs[i] = imgs[i]
                targets[i] = masks[i]
            return inputs, targets, names
        else:
            names = [x[1] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            for i in range(num_imgs):
                inputs[i] = imgs[i]
            return inputs, names

