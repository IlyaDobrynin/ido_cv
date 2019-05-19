# -*- coding: utf-8 -*-
"""
Module implements functions for work with dataset

"""
import os
import cv2
import numpy as np
import torch
from albumentations.torch.functional import img_to_tensor
from albumentations import Compose
from albumentations import Resize
from torch.utils.data import Dataset

from ...utils.image_utils import pad
from ...utils.image_utils import resize
from ...utils.image_utils import resize_image
from ...utils.image_utils import draw_images
from ...utils.image_utils import convert_multilabel_mask


def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[1, row, :] = const
    image_tensor[2] = image_tensor[0] * image_tensor[1]
    return image_tensor


class BinSegDataset(Dataset):
    """ Class describes current dataset
    """
    
    def __init__(self, initial_size: tuple, model_input_size: int, train: bool,
                 add_depth: bool = False, data_path: str = None, data_file: np.ndarray = None,
                 augmentations=None, show_sample: bool = True, **kwargs):

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

        self.initial_size = initial_size

        if not model_input_size:
            self.model_input_size = initial_size
        else:
            self.model_input_size = model_input_size
            
        self.augmentations = augmentations
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
        image = resize_image(image=image, size=self.initial_size)

        if self.train:
            if self.from_path:
                mask = cv2.imread(os.path.join(self.data_path, r'masks/{}'.format(fname)), 0)
            else:
                mask = self.data_file[self.data_file['names'] == fname]['masks'].values[0]
            mask = resize_image(mask, size=self.initial_size, interpolation=cv2.INTER_NEAREST)
            return self._get_trainset(image, mask, fname)
        else:
            return self._get_testset(image, fname)
    
    def _get_trainset(self, image, mask, name):
        image = pad(image, mode='edge')
        image = resize(image, size=self.model_input_size)
        mask = pad(mask)
        mask = resize(mask, size=self.model_input_size, interpolation=cv2.INTER_NEAREST)
        if self.augmentations:
            data = {'image': image, 'mask': mask}
            augmented = self.augmentations(**data)
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
            viz_mask = np.moveaxis(mask.data.numpy(), 0, -1)
            draw_images([viz_image, viz_mask])

        return image, mask, str(name)
    
    def _get_testset(self, image, name):
        image = pad(image)
        image = resize(image, size=self.model_input_size)
        if self.augmentations:
            data = {'image': image}
            augmented = self.augmentations(**data)
            image = augmented['image']
        if self.add_depth:
            image = add_depth_channels(img_to_tensor(image))

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if self.show_sample:
            viz_image = np.moveaxis(image.data.numpy(), 0, -1)
            draw_images([viz_image])

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
        h = w = self.model_input_size
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
            names = [x[1] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            for i in range(num_imgs):
                inputs[i] = imgs[i]
            return inputs, names


class MultSegDataset(Dataset):
    """ Class describes current dataset
    """
    
    def __init__(self, initial_size: tuple, model_input_size: int, train: bool,
                 add_depth: bool = False, data_path: str = None, data_file: np.ndarray = None,
                 augmentations=None, show_sample: bool = False, **kwargs):

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

        self.initial_size = initial_size
        if not model_input_size:
            self.model_input_size = initial_size
        else:
            self.model_input_size = model_input_size
        
        self.colors = kwargs['label_colors']
        self.augmentations = augmentations
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
        # ToDo remove hardcode
        # image = pad(image, mode='reflect')
        # image = resize(image, size=self.model_input_size)
        # image = resize_image(image, size=self.model_input_size)
        # # mask = pad(mask, mode='reflect')
        # # mask = resize(mask, size=self.model_input_size, interpolation=cv2.INTER_NEAREST)
        # mask = resize_image(mask, size=self.model_input_size, interpolation=cv2.INTER_NEAREST)
        mask = convert_multilabel_mask(mask, label_colors=self.colors, how='rgb2class')
        data = {'image': image, 'mask': mask}
        resized = Compose([
            Resize(width=self.model_input_size, height=self.model_input_size)
        ], p=1)(**data)
        image, mask = resized['image'], resized['mask']

        # mask = delete_small_instances(mask, obj_size=10)
        
        if self.augmentations:
            data = {'image': image, 'mask': mask}
            augmented = self.augmentations(**data)
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
        image = pad(image)
        image = resize(image, size=self.model_input_size)
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
        h = w = self.model_input_size
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

