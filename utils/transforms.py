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
