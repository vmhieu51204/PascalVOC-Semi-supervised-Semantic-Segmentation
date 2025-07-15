import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

IMAGE_SIZE = (320, 320)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_albu_transform(): 
    return A.Compose([
    A.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0.5, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
    A.Normalize(MEAN, STD),
    ToTensorV2(),
])

def get_val_transform(): 
    return A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(MEAN, STD),
    ToTensorV2(),
])

class AlbuTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, label = None):
        img = np.array(img)
        if label is not None:
            label = np.array(label)
            augmented = self.transform(image=img, mask=label)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.transform(image=img)
            return augmented['image']