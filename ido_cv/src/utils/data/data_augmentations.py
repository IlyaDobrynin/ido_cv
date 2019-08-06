# -*- coding: utf-8 -*-
"""
Module implement data loader constructors

"""
import cv2
from abc import ABC, abstractmethod
from albumentations import (Compose,
                            RandomBrightness,
                            ShiftScaleRotate,
                            OneOf,
                            PadIfNeeded,
                            Resize,
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


class Augmentations(ABC):
    """
    Abstract class Class implements model train-time and test-time augmentations
    Arguments:
        mode:       type of augmentations:
                        - train_time_augs
                        - common_augs
        p:          probability of augmentations accept
    """

    @abstractmethod
    def __train_time_transform(self, *args, **kwargs):
        """ Method to make transformations of train images
        :return:
        """
        # return Compose([
        #     RandomBrightness(limit=(0.05), p=0.5),
        #     RandomContrast(limit=(-0.05, 0.05), p=0.5),
        #     OneOf([
        #         JpegCompression(
        #             quality_lower=40,
        #             quality_upper=100
        #         ),
        #     #     IAAAdditiveGaussianNoise(scale=(0.05 * 255, 0.1 * 255), p=0.5),
        #     #     # GaussNoise(p=0.5),
        #     ], p=0.5),
        #     # # OneOf([
        #     #     # MotionBlur(blur_limit=3, p=0.5),
        #     #     # MedianBlur(blur_limit=3, p=0.5)
        #     # # ], p=0.5),
        #     Cutout(
        #         num_holes=500,
        #         max_h_size=2,
        #         max_w_size=2,
        #         p=0.5
        #     ),
        #     # ElasticTransform(sigma=1, alpha_affine=1, p=1),
        #     # OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1),
        #     ShiftScaleRotate(shift_limit=0.,
        #                      scale_limit=(-0.2, 0.2),
        #                      rotate_limit=5,
        #                      border_mode=cv2.BORDER_CONSTANT,
        #                      interpolation=cv2.INTER_AREA,
        #                      p=0.5),
        # ], p=1)
        pass

    def __common_augs(self, *args, **kwargs):
        """ Method to make augmentations common for all images

        """
        # return Compose([
        #     PadIfNeeded(
        #         min_height=size,
        #         min_width=size,
        #         border_mode=cv2.BORDER_CONSTANT
        #     ),
        #     Resize(
        #         width=size,
        #         height=size
        #     )
        # ], p=1)
        pass

    # @property
    def get_augmentations(self, mode: str, *args, **kwargs):
        if mode == 'train_time_augs':
            return self.__train_time_transform(*args, **kwargs)
        elif mode == 'common_augs':
            return self.__common_augs(*args, **kwargs)
        else:
            raise ValueError(
                f"Wrong mode parameter: {mode}. "
                f"Should be 'train_time_augs' or 'common_augs'."
            )
