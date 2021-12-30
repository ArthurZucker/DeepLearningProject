import numpy as np
import torch

#data augmentation libraries
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def transform_global1(img_size, global_crops_scale):
    
    # Here, A.RandomBrightnessContrast and A.HueSaturationValue replace the ColorJitter  
    # because the torchvision.transform implementation of ColorJitter is different from the Albumentation one 
    transform = A.Compose(
        [
            A.RandomResizedCrop (height=img_size, width=img_size, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=int(0.1 * 180),
                                 sat_shift_limit=int(0.2 * 255),
                                 val_shift_limit=0, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur( sigma_limit=[0.1, 0.2], p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
    )

    return transform

def transform_global2(img_size, global_crops_scale):
    
    # Here, A.RandomBrightnessContrast and A.HueSaturationValue replace the ColorJitter  
    # because the torchvision.transform implementation of ColorJitter is different from the Albumentation one 
    transform = A.Compose(
        [
            A.RandomResizedCrop (height=img_size, width=img_size, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=int(0.1 * 180),
                                 sat_shift_limit=int(0.2 * 255),
                                 val_shift_limit=0, p=0.8),
            A.ToGray(p=0.2),
            A.Solarize (p=0.2),
            A.GaussianBlur( sigma_limit=[0.1, 0.2], p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
    )

    return transform

def transform_local(img_size, local_crops_scale):
    
    # Here, A.RandomBrightnessContrast and A.HueSaturationValue replace the ColorJitter  
    # because the torchvision.transform implementation of ColorJitter is different from the Albumentation one 
    transform = A.Compose(
        [
            A.RandomResizedCrop (height=img_size, width=img_size, scale=local_crops_scale, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=int(0.1 * 180),
                                 sat_shift_limit=int(0.2 * 255),
                                 val_shift_limit=0, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur( sigma_limit=[0.1, 0.2], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
    )

    return transform
