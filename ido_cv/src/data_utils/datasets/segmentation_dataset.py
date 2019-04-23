# -*- coding: utf-8 -*-
"""
Module implements functions for work with dataset

"""
import os
import cv2
import numpy as np
import torch
from albumentations.torch.functional import img_to_tensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from ... import allowed_parameters
from ...utils.image_utils import pad
from ...utils.image_utils import resize
from ...utils.image_utils import resize_image


def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[1, row, :] = const
    image_tensor[2] = image_tensor[0] * image_tensor[1]
    return image_tensor


class BinSegDataset(Dataset):
    """ Class describes current dataset
    """
    
    def __init__(self, train, initial_size, model_input_size, add_depth=False, data_path=None,
                 data_file=None, augmentations=None, show_sample=False):

        self.train = train

        if data_path is not None:
            self.data_path = data_path
            self.file_names = os.listdir(os.path.join(self.data_path, 'images'))
            self.from_path = True
        elif data_file is not None:
            self.file_names = data_file['names'].values.tolist()
            self.data_file = data_file
            self.from_path = False

        self.initial_size = initial_size

        self.show_sample = show_sample

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
            self._draw_sample(viz_image, viz_mask)

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
            self._draw_sample(viz_image)

        image = img_to_tensor(image)
        return image, str(name)
    
    @staticmethod
    def _draw_sample(image, mask=None):
        # print(np.max(image), np.max(mask))
        import matplotlib.pyplot as plt
        fig = plt.figure()
        if mask is not None:
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(image)
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(mask[:, :, 0])
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(image)
        plt.show()

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
    
    def __init__(self, train, initial_size, model_input_size, add_depth=False, data_path=None,
                 data_file=None, augmentations=None):

        self.train = train

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
        
        self.colors = allowed_parameters.SEG_MULTI_COLORS
        self.augmentations = augmentations
        self.add_depth = add_depth
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        fname = self.file_names[idx]
        if self.from_path:
            image = cv2.imread(os.path.join(self.data_path, fr'images/{fname}'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = self.data_file[self.data_file['names'] == fname]['images'].values[0]
        image = resize_image(image=image, size=self.initial_size)

        if self.train:
            if self.from_path:
                mask = cv2.imread(os.path.join(self.data_path, f'masks/{fname}'))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                mask = self.data_file[self.data_file['names'] == fname]['masks'].values[0]
            mask = resize_image(image=mask, size=self.initial_size, interpolation=cv2.INTER_NEAREST)
            return self._get_trainset(image, mask, fname)
        else:
            return self._get_testset(image, fname)
    
    def _get_trainset(self, image, mask, name):
        image = pad(image)
        image = resize(image, size=self.model_input_size)
        mask = pad(mask)
        mask = resize(mask, size=self.model_input_size, interpolation=cv2.INTER_NEAREST)
        mask = self.convert_multilabel_mask(mask, how='rgb2class', n_classes=11)
        
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
        
        viz_image = np.moveaxis(image.data.numpy(), 0, -1)
        viz_mask = np.moveaxis(mask.data.numpy(), 0, -1)[:, :, 0]
        # self._draw_sample(viz_image, viz_mask)
        
        return image, mask, str(name)
    
    def _get_testset(self, image, name):
        image = pad(image)
        image = resize(image, size=self.model_input_size)
        if self.add_depth:
            image = add_depth_channels(img_to_tensor(image))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # self._draw_sample(image)
        
        image = img_to_tensor(image)
        return image, str(name)
    
    @staticmethod
    def _draw_sample(image, mask=None):
        fig = plt.figure()
        if mask is not None:
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(image)
            if mask.shape[-1] == 1:
                ax = fig.add_subplot(1, 2, 2)
                ax.imshow(mask[:, :, 0])
            else:
                ax = fig.add_subplot(1, 2, 2)
                ax.imshow(mask)
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(image)
            
        plt.show()

    @staticmethod
    def convert_multilabel_mask(mask, how='rgb2class', n_classes=2, threshold=0.99):
        """ Function for multilabel mask convertation

        :param mask:
        :param how:
        :return:
        """
        colors = allowed_parameters.SEG_MULTI_COLORS
        if how == 'rgb2class':
            out_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
            for cls in range(n_classes):
                if cls == 11:
                    continue
                matching = np.all(mask == colors[cls], axis=-1)
                out_mask[matching] = cls + 1
    
        elif how == 'class2rgb':
            out_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for ch in range(mask.shape[-1]):
                if ch == 0:
                    continue
                matching = (mask[:, :, ch] > threshold)
                out_mask[matching, :] = colors[ch - 1]
        else:
            raise ValueError(
                f"Wrong parameter how: {how}. Should be 'rgb2class or class2rgb."
            )
        return out_mask
    
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

