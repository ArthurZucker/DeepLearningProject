import cv2 

import albumentations as A
from albumentations.pytorch import ToTensorV2

class BarlowTwinsTransform(object):
    """TODO docstring
    """
    def __init__(self, img_size) -> None:
        
        self.img_size = img_size

    def __call__(self, image):
        
        base_transform = A.Compose(
            [A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.RandomResizedCrop(
                height=self.img_size[0],
                width =self.img_size[1],
                scale=(0.08, 1.0),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(
                hue_shift_limit=int(0.1 * 180),
                sat_shift_limit=int(0.2 * 255),
                val_shift_limit=0,
                p=0.8,
            ),
            A.ToGray(p=0.2),]
        )
        transform1 = A.Compose(
            [A.Solarize(p=0.0),
            A.GaussianBlur(sigma_limit=[0.1, 0.2], p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
            ]
        )

        transform2 = A.Compose(
            [A.Solarize(p=0.2),
            A.GaussianBlur(sigma_limit=[0.1, 0.2], p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
            ]
        )

        base_aug_image1 = base_transform(image=image)["image"]
        base_aug_image2 = base_transform(image=image)["image"]

        aug_image1 = transform1(image=base_aug_image1)["image"]
        aug_image2 = transform2(image=base_aug_image2)["image"]

        return aug_image1, aug_image2


class BarlowTwinsTransformEval(object):
    """TODO docstring
    """
    def __init__(self, img_size) -> None:
        
        self.img_size = img_size

    def __call__(self, image):
        
        transform = A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
            ]
        )

        aug_image = transform(image=image)["image"]

        return aug_image



class DinoTransforms(object):
    """
    Transforms for dino model:
    - One transform for the first global crop
    - One transform for the second global crop
    - One transform to apply on each local crop
    returns a list of transformed crops of the input image
    """
    def __init__(self, img_size, n_crops, n_global_crops, local_crops_scale, global_crops_scale) -> None:
        
        self.img_size = img_size
        self.n_crops = n_crops
        self.n_global_crops = n_global_crops
        self.local_crops_scale = local_crops_scale
        self.global_crops_scale = global_crops_scale

    def __call__(self, image):
        
        transform_global1 = A.Compose(
        [
            A.RandomResizedCrop (height=self.img_size[0], width=self.img_size[1], scale=self.global_crops_scale, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(
                                hue_shift_limit=int(0.1 * 180),
                                sat_shift_limit=int(0.2 * 255),
                                val_shift_limit=0, p=0.8
                                ),
            A.ToGray(p=0.2),
            A.GaussianBlur(sigma_limit=[0.1, 0.2], p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        )

        transform_global2 = A.Compose(
        [
            A.RandomResizedCrop (height=self.img_size[0], width=self.img_size[1], scale=self.global_crops_scale, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(
                                hue_shift_limit=int(0.1 * 180),
                                sat_shift_limit=int(0.2 * 255),
                                val_shift_limit=0, p=0.8
                                ),
            A.ToGray(p=0.2),
            A.Solarize (p=0.2),
            A.GaussianBlur(sigma_limit=[0.1, 0.2], p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        )

        transform_local = A.Compose(
        [
            A.RandomResizedCrop (height=self.img_size[0], width=self.img_size[1], scale=self.local_crops_scale, interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(
                                hue_shift_limit=int(0.1 * 180),
                                sat_shift_limit=int(0.2 * 255),
                                val_shift_limit=0, p=0.8
                                ),
            A.ToGray(p=0.2),
            A.GaussianBlur(sigma_limit=[0.1, 0.2], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        )

        aug_images = []
        
        aug_images.append(transform_global1(image=image)["image"])
        aug_images.append(transform_global2(image=image)["image"])
        for _ in range(self.n_crops - self.n_global_crops):
            aug_images.append(transform_local(image=image)["image"])

        return aug_images