import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from albumentations.torch.functional import img_to_tensor
from ...utils.image_utils import draw_images


class OCRDataset(Dataset):
    """Digits dataset."""
    def __init__(
            self,
            train: bool,
            data_path: str = None,
            data_file: pd.DataFrame = None,
            common_augs=None,
            train_time_augs=None,
            show_sample=False,
            **kwargs
    ):
        self.train = train
        self.show_sample = show_sample
        if data_path is not None:
            self.data_path = data_path
            images_path = os.path.join(self.data_path, 'images')
            if os.path.exists(images_path):
                self.images_path = images_path
            else:
                raise ValueError(
                    f"Wrong data_path: this folder doesn't have images inside: {self.data_path}"
                )
            if self.train:
                labels_path = os.path.join(self.data_path, r'labels.csv')
                if os.path.exists(labels_path):
                    self.labels_path = labels_path
                else:
                    raise ValueError(
                        f"Wrong data_path: this folder doesn't have labels inside: {self.labels_path}"
                    )

            self.file_names = os.listdir(self.images_path)
            self.from_path = True
        elif data_file is not None:
            self.file_names = data_file['names'].values.tolist()
            self.data_file = data_file
            self.from_path = False
        else:
            raise ValueError(
                f"data_path or data_file should be provided"
            )

        # self.img_names = []
        self.targets = []
        self.common_augs = common_augs
        self.train_time_augs = train_time_augs
        characters = kwargs['alphabet']
        self.alphabet = {characters[i]: i for i in range(len(characters))}

    def _encode(self, label: str):
        """ Function makes from the given string list of numbers
        corresponded to the given character. For example:
        alphabet = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        label = 'dabcda'
        out_label = [4, 1, 2, 3, 4, 1]

        :param label:   String with given label
        :return:
        """
        out_label = []
        for i in range(len(label)):
            char = label[i]
            out_label.append(self.alphabet[char])
        return out_label

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        if self.from_path:
            image = cv2.imread(os.path.join(self.images_path, fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = self.data_file[self.data_file['names'] == fname]['images'].values[0]
        if self.train:
            if self.from_path:
                labels = pd.read_csv(
                    self.labels_path, sep=';', header=None, index_col=0, names=['labels']
                )
                label = labels[labels.index == fname]['labels'].values[0]
            else:
                label = self.data_file[self.data_file['names'] == fname]['labels'].values[0]
            label = label.strip().replace(" ", "")
            return self._get_trainset(image, label, fname)
        else:
            return self._get_testset(image, fname)

    def _get_trainset(self, image, label, name):
        """ Function returns train image, label and name

        :param image:   Input image, np.ndarray
        :param label:   Input label, str
        :param name:    Imput name, str
        :return:
        """
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
        label_ = self._encode(label)
        if self.show_sample:
            print(label)
            viz_image = np.moveaxis(image.data.numpy(), 0, -1)
            draw_images([viz_image])

        label_ = torch.IntTensor([int(i) for i in label_])
        return image, label_, str(name)

    def _get_testset(self, image, name):
        """ Function returns test image and name

        :param image:   Input image, np.ndarray
        :param name:    Imput name, str
        :return:
        """
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
        """Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        """
        transposed_data = list(zip(*batch))
        images = torch.stack(transposed_data[0], 0)
        # ToDo add checks
        # h = w = imgs[0].shape[1]
        if self.train:
            targets = torch.cat(transposed_data[1], 0)
            # print('collate_fn', targets)
            target_lengths = torch.IntTensor([len(i) for i in transposed_data[1]])
            # print('collate_fn', target_lengths)
            names = transposed_data[2]
            return images, targets, target_lengths, names
        else:
            names = transposed_data[1]
            return images, names

