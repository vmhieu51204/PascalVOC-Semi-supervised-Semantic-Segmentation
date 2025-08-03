import numpy as np
import os
from torchvision.transforms import ToTensor,Compose
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch.nn.functional as F
import torchvision.models as models
from functools import reduce
import torch.optim as optim
from torchvision import transforms
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import SBDataset, VOCSegmentation
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp

from utils.utils import *
from data.datasets  import VOCDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (320, 320)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 21
BATCH_SIZE = 12

def main(root, models):
    val_transform = A.Compose([
    A.Resize(*IMAGE_SIZE),
    A.Normalize(MEAN, STD),
    ToTensorV2(),
    ])
    valset = VOCDataset(root, "val", transform=val_transform)
    valoader = DataLoader(valset,batch_size=12, shuffle=False, num_workers=2,drop_last=False)
    v_miou_mt = mIoU(num_classes=21)
    v_miou_gan = mIoU(num_classes=21)
    val_pa_mt = 0
    val_pa_gan = 0
    mt = torch.load(os.join(models, "mt_eff.pt"), weights_only=False)
    gan = torch.load(os.join(models, "experiment_01.pt"), weights_only=False)
    mt.to(DEVICE)
    gan.to(DEVICE)
    mt.eval()
    gan.eval()
    with torch.no_grad():
        print("Evaluating Mean Teacher model...")
    for i, data in enumerate(tqdm(valoader)):
        image, mask = data
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        output_s, output_t = mt(image)
        val_pa_mt += pixel_accuracy(output_t, mask)
        v_miou_mt.add_batch(torch.argmax(output_t, dim=1).cpu().numpy(), mask.cpu().numpy())
    print("Pixel accuracy: ", val_pa_mt / len(valoader))
    print("Mean teacher mIoU: ", v_miou_mt.evaluate())

    with torch.no_grad():
        print("Evaluating GAN model...")
        for i, data in enumerate(tqdm(valoader)):
            image, mask = data
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            
            output = gan(image)
            if isinstance(output, tuple):
                output = output[0]

            val_pa_gan += pixel_accuracy(output, mask)
            v_miou_gan.add_batch(torch.argmax(output, dim=1).cpu().numpy(), mask.cpu().numpy())
    print("Pixel accuracy: ", val_pa_gan / len(valoader))
    print("GAN mIoU: ", v_miou_gan.evaluate())

    miou_s4, pa_s4 = val_mt_fused(gan, mt, valoader)
    print("Evaluating s4GAN model...")
    print("Pixel accuracy: ", pa_s4)
    print("s4GAN mIoU: ",miou_s4)

