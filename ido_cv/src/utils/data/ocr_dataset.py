# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
from mts_cv.src.utils.image_utils import draw_images


class OCRDataset(Dataset):
    """ Dataset loader for OCR task
    Class reads files from folder or dataframe.
    Folder should contain 'images' subfolder with images to load
    and file labels.csv with labels.

    Arguments:
        train:              Flag to show is the dataset train or not
        alphabet_dict:      Dictionary for coding alphabet
        data_path:          Path to data folder
        data_file:          DataFrame with training images and labels
        common_augs:        Augmentations for images before model input
                            (for both train and inference process)
        train_time_augs:    Augmentations for images before training
                            (only for train process)
        show_sample:        Flag to show the samplesof images before training

    """
    def __init__(
            self,
            train:          bool,
            alphabet_dict:  dict,
            data_path:      str = None,
            data_file:      pd.DataFrame = None,
            common_augs=None,
            train_time_augs=None,
            show_sample:    bool = False
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
                        f"Wrong data_path: "
                        f"this folder doesn't have labels inside: {self.labels_path}"
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

        self.targets = []
        self.common_augs = common_augs
        self.train_time_augs = train_time_augs
        self.alphabet_dict = alphabet_dict

    def _encode(self, label: str) -> list:
        """ Function makes from the given string list of numbers
        corresponded to the given character. For example:
        alphabet = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        label = 'dabcda'
        out_label = [4, 1, 2, 3, 4, 1]
        Also replace common symbols for russian and english to russian

        :param label:   String with given label
        :return: list of encoded label
        """
        out_label = []
        for i in range(len(label)):
            char = label[i]
            ru_common = [ch for ch in 'аеорсухАВЕКМНОРСТХ']
            en_common = [ch for ch in 'aeopcyxABEKMHOPCTX']
            common_char_dict = {ru: [ru, en] for ru, en in zip(ru_common, en_common)}
            for ru_char, ru_eng_chars in common_char_dict.items():
                if char in ru_eng_chars:
                    char = ru_char
            out_label.append(self.alphabet_dict[char])
        return out_label

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> tuple:
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
            label = str(label).strip().replace(" ", "")
            return self._get_trainset(image, label, fname)
        else:
            return self._get_testset(image, fname)

    def _get_trainset(self, image: np.ndarray, label: str, name: str):
        """ Function returns train image, label and name

        :param image:   Input image, np.ndarray
        :param label:   Input label, str
        :param name:    Input name, str
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

    def _get_testset(self, image: np.ndarray, name: str):
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
        """Function to form batches for training and inference

        """
        transposed_data = list(zip(*batch))
        images = torch.stack(transposed_data[0], 0)
        if self.train:
            targets = torch.cat(transposed_data[1], 0)
            target_lengths = torch.IntTensor([len(i) for i in transposed_data[1]])
            names = transposed_data[2]
            return images, targets, target_lengths, names
        else:
            names = transposed_data[1]
            return images, names
