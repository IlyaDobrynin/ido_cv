# -*- coding: utf-8 -*-
"""
Module implements functions for work with dataset

"""
from __future__ import print_function
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
from mts_cv.src.utils.data.encoder_alt import DataEncoder
from mts_cv.src.utils.image_utils import pad
from mts_cv.src.utils.image_utils import resize


class RetinaDataset(Dataset):
    def __init__(self, root, labels_file, train, initial_size, model_input_size,
                 augmentations=None):
        """
        Args:
          root: (str) ditectory to images.
          labels_file: (str) path to index file.
          train: (boolean) train or test.
          input_size: (int) model input size.
        """
        self.root = os.path.join(root, 'images')
        self.train = train
        self.initial_size = initial_size
        self.model_input_size = model_input_size
        self.augmentations = augmentations
        self.encoder = DataEncoder(input_size=model_input_size)
        
        if self.train:
            self.fnames = []
            self.boxes = []
            self.labels = []
            
            with open(labels_file) as f:
                lines = f.readlines()
                self.num_samples = len(lines)
            
            for line in lines:
                splited = line.strip().split()
                self.fnames.append(splited[0])
                num_boxes = (len(splited) - 1) // 5
                box = []
                label = []
                for i in range(num_boxes):
                    xmin = splited[1 + 5 * i]
                    ymin = splited[2 + 5 * i]
                    xmax = splited[3 + 5 * i]
                    ymax = splited[4 + 5 * i]
                    c = splited[5 + 5 * i]
                    box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                    label.append(int(c))
                box = np.asarray(box, dtype=np.float32)
                label = np.asarray(label, dtype=np.int64)
                self.boxes.append(torch.from_numpy(box))
                self.labels.append(torch.from_numpy(label))
        else:
            self.fnames = os.listdir(self.root)
            self.num_samples = len(self.fnames)
    
    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        # img = Image.open(os.path.join(self.root, fname))
        img = cv2.imread(os.path.join(self.root, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        # img = np.array(img)
        # print('dataset', img.shape)
        img = self._resize_initial_image(image=img)
        if self.train:
            boxes = self.boxes[idx]
            labels = self.labels[idx]
            return self._get_trainset(image=img, boxes=boxes, labels=labels, name=fname)
        else:
            return self._get_testset(image=img, name=fname)
    
    def _resize_initial_image(self, image):
        h_i, w_i = image.shape[0], image.shape[1]
        h, w = self.initial_size
        if h == h_i and w == w_i:
            return image
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _get_trainset(self, image, boxes, labels, name):
        image, boxes = pad(image, boxes)
        image, boxes = resize(image, boxes, self.model_input_size)
        if self.augmentations is not None:
            data = {'image': image, 'bboxes': boxes}
            augmented = self.augmentations(**data)
            image = augmented['image']
            boxes = augmented['bboxes']
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # self._draw_image_boxes(image=image, boxes=boxes)
        
        image = img_to_tensor(image)
        return image, boxes, labels, name
    
    def _get_testset(self, image, name):
        image = pad(image)
        image = resize(image, size=self.model_input_size)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # self._draw_image_boxes(image=image)
        
        image = img_to_tensor(image)
        
        return image, name
    
    @staticmethod
    def _draw_image_boxes(image, boxes=None):
        import matplotlib.pyplot as plt
        draw = image.copy()
        if boxes is not None:
            for box in boxes:
                draw = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        if draw.shape[-1] == 3:
            plt.imshow(draw)
        else:
            plt.imshow(draw[:, :, 0])
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
            boxes = [x[1] for x in batch]
            labels = [x[2] for x in batch]
            names = [x[3] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            loc_targets = []
            cls_targets = []
            for i in range(num_imgs):
                inputs[i] = imgs[i]
                loc_target, cls_target = self.encoder.encode(boxes[i], labels[i])
                loc_targets.append(loc_target)
                cls_targets.append(cls_target)
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets), names
        else:
            names = [x[1] for x in batch]
            num_imgs = len(imgs)
            inputs = torch.zeros(num_imgs, 3, h, w)
            for i in range(num_imgs):
                inputs[i] = imgs[i]
            return inputs, names
    
    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    root = '/home/ilyado/Programming/work_mts/dates_recognition/data/dates/train/images'
    labels = '/home/ilyado/Programming/work_mts/dates_recognition/data/dates/train/dates.txt'