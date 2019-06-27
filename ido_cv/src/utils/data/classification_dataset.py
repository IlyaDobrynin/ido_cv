# -*- coding: utf-8 -*-
"""
Module implements functions for work with classification dataset

"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset
from mts_cv.src.utils.image_utils import draw_images


class ClassifyDataset(Dataset):
    """ Class describes current dataset
    """

    def __init__(self, train: bool, data_path: str = None, data_file: pd.DataFrame = None,
                 common_augs=None, train_time_augs=None, show_sample=False, **kwargs):

        self.train = train

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

        self.show_sample = show_sample
        self.common_augs = common_augs
        self.train_time_augs = train_time_augs

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
                labels = pd.read_csv(os.path.join(self.data_path, r'labels.csv'), sep=';',
                                     header=None, index_col=0, names=['labels'])
                label = int(labels[labels.index == fname]['labels'].values[0])
            else:
                label = self.data_file[self.data_file['names'] == fname]['labels'].values[0]
            return self._get_trainset(image, label, fname)
        else:
            return self._get_testset(image, fname)

    def _get_trainset(self, image, label, name):
        if self.common_augs:
            data = {'image': image}
            common_augs = self.common_augs(**data)
            image = common_augs['image']
        if self.train_time_augs:
            data = {'image': image}
            augmented = self.train_time_augs(**data)
            image = augmented['image']

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image = img_to_tensor(image)

        if self.show_sample:
            viz_image = np.moveaxis(image.data.numpy(), 0, -1)
            draw_images([viz_image])

        return image, label, str(name)

    def _get_testset(self, image, name):
        if self.common_augs:
            data = {'image': image}
            common_augs = self.common_augs(**data)
            image = common_augs['image']
        if self.train_time_augs:
            data = {'image': image}
            augmented = self.train_time_augs(**data)
            image = augmented['image']

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image = img_to_tensor(image)

        if self.show_sample:
            viz_image = np.moveaxis(image.data.numpy(), 0, -1)
            draw_images([viz_image])
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
        h = w = imgs[0].shape[1]
        if self.train:
            labels = np.expand_dims(np.asarray([x[1] for x in batch], dtype=np.float32), axis=-1)
            # print('collate.labels', labels)
            names = [x[2] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            for i in range(num_imgs):
                inputs[i] = imgs[i]
            targets = torch.from_numpy(labels)
            return inputs, targets, names
        else:
            names = [x[1] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            for i in range(num_imgs):
                inputs[i] = imgs[i]
            return inputs, names
