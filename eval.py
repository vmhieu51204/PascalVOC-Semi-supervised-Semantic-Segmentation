import numpy as np
import os
from torchvision.transforms import ToTensor,Compose, ToPILImage
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
import matplotlib.pyplot as plt
import random

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
    random_id = 800
    if 'valset' not in globals():
        print("Error: `valset` is not defined. Please ensure the dataset loading code from your setup has been run.")
    else:
        random_idx = random_id
        input_tensor, gt_mask_tensor = valset[random_idx] 

        original_image_denormalized_tensor = denormalize_image_tensor(input_tensor, MEAN, STD)
        original_image_pil = ToPILImage()(original_image_denormalized_tensor)

        model_input_batch = input_tensor.unsqueeze(0).to(DEVICE)

        if 'pascal_palette_invert' not in globals():
            print("Error: `pascal_palette_invert` function is not defined.")
            exit()
        flat_pascal_palette = pascal_palette_invert()

        gt_mask_rgb_pil = tensor_mask_to_rgb_pil(gt_mask_tensor, flat_pascal_palette)
        
        model_predictions_rgb = {}
        prediction_order = [] # To maintain a specific order for plotting

        # Define models and their names in the desired plotting order
        models_to_process = [
            ("Base", base),
            ("ST", st),
            ("ST++", stpp),
            ("GAN ", gan)
        ]

        for name, model_obj in models_to_process:
            print(f"Generating prediction for {name}...")
            pred_mask_tensor = get_model_prediction_single_image(model_obj, model_input_batch)
            model_predictions_rgb[name] = tensor_mask_to_rgb_pil(pred_mask_tensor, flat_pascal_palette)
            prediction_order.append(name)

        # Prediction from Mean Teacher's teacher model
        mt_teacher_name = "Mean Teacher (Teacher)"
        print(f"Generating prediction for {mt_teacher_name}...")
        mt_teacher_pred_tensor = get_mt_teacher_prediction_single_image(mt, model_input_batch)
        model_predictions_rgb[mt_teacher_name] = tensor_mask_to_rgb_pil(mt_teacher_pred_tensor, flat_pascal_palette)
        prediction_order.append(mt_teacher_name)
        
        # Prediction for the special s4gan fused model
        s4gan_fused_name = "S4GAN"
        print(f"Generating prediction for {s4gan_fused_name}...")
        s4gan_fused_pred_tensor = get_s4gan_fused_prediction_single_image(gan, mt, model_input_batch)
        model_predictions_rgb[s4gan_fused_name] = tensor_mask_to_rgb_pil(s4gan_fused_pred_tensor, flat_pascal_palette)
        prediction_order.append(s4gan_fused_name)

        # 4. Plotting in a 3x3 grid
        # Total items: Original, GT (2) + number of model predictions (7) = 9 items.
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10)) # 3 rows, 3 columns
        ax_flat = axes.ravel() # Flatten the 2D array of axes for easy indexing
        
        # Plot Model Predictions
        # Start plotting predictions from the 3rd subplot (index 2)
        for i, name in enumerate(prediction_order):
            plot_idx = i  # ax_flat[0] and ax_flat[1] are already used
            if plot_idx < 9: # Ensure we are within the 3x3 grid
                ax_flat[plot_idx].imshow(model_predictions_rgb[name])
                ax_flat[plot_idx].set_title(name)
                ax_flat[plot_idx].axis('off')
            else: # Should not happen if we have 7 predictions + 2 images for a 3x3 grid
                print(f"Warning: More plots than expected for a 3x3 grid. Skipping '{name}'.")
                
        # Turn off axes for any remaining empty subplots (if fewer than 9 items total)
        # This loop handles cases where you might have fewer than 7 model predictions later.
        # For the current setup (2 fixed + 7 predictions = 9), this loop won't do anything extra
        # as all 9 cells will be filled.
        for i in range(len(prediction_order), 6): # from the next empty cell up to 8
            ax_flat[i].axis('off')
            
        plt.tight_layout()
        # --- Save the plot ---
        plot_filename = f"segmentation_{random_id}.png"
        try:
            plt.savefig(plot_filename, dpi=600, bbox_inches='tight') # dpi for resolution, bbox_inches for tight layout
            print(f"Plot saved successfully as {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        # --- End save the plot ---
        plt.show()

