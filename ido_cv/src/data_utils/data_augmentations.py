# -*- coding: utf-8 -*-
"""
Module implement data loader constructors

"""
import cv2
from albumentations import (Compose,
                            RandomBrightness,
                            ShiftScaleRotate,
                            OneOf,
                            ElasticTransform,
                            OpticalDistortion,
                            Cutout,
                            GaussNoise,
                            Blur,
                            MedianBlur,
                            MotionBlur,
                            IAAAdditiveGaussianNoise,
                            JpegCompression,
                            RandomContrast)


class Augmentations:
    """
    Class implements model train-time and test-time augmentations
    Arguments:
        mode - type of augmentations (train, valid or test)
        p - probability of augmentations accept
    """
    def __init__(self, train, p=1):
        self.p = p
        if train:
            self.transform = self.train_transform()
        else:
            self.transform = self.test_transform()
            
    def train_transform(self):
        """ Method to make transformations of train images
        :return:
        """
        # return OneOf([
        #     GaussNoise(p=0.5),
        #     Blur(blur_limit=4, p=0.5),
        #     JpegCompression(quality_lower=50, quality_upper=90, p=0.5),
        #     Cutout(max_h_size=1, max_w_size=1, num_holes=350, p=0.5)
        # ])
        return Compose([
            RandomBrightness(limit=(0.05), p=0.5),
            RandomContrast(limit=(-0.05, 0.05), p=0.5),
            OneOf([
                JpegCompression(quality_lower=70, quality_upper=100),
                IAAAdditiveGaussianNoise(scale=(0.05 * 255, 0.1 * 255), p=0.5),
                # GaussNoise(p=0.5),
            ], p=0.5),
            # OneOf([
                # MotionBlur(blur_limit=3, p=0.5),
                # MedianBlur(blur_limit=3, p=0.5)
            # ], p=0.5),
            Cutout(num_holes=500, max_h_size=2, max_w_size=2, p=0.5),
            # ElasticTransform(sigma=1, alpha_affine=1, p=1),
            # OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1),
            ShiftScaleRotate(shift_limit=0., scale_limit=(-0.2, 0), rotate_limit=10, border_mode=cv2.BORDER_DEFAULT, p=0.5),
        ], p=self.p)

    def val_transform(self):
        """ Method to make transformations of valid images
        :return:
        """
        return Compose([
        ], p=self.p)

    def test_transform(self):
        """ Method to make transformations of test images
        :return:
        """
        return Compose([
        ], p=self.p)