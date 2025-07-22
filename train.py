#!pip install -q segmentation-models-pytorch
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
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import SBDataset, VOCSegmentation
from data.datasets import make_datasets
from models.models import Dis, MeanTeacherNetwork
from utils.utils import val

import segmentation_models_pytorch as smp
import argparse

IMAGE_SIZE = (320, 320)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

config = {
    "prefix": "train_",
    "lam_adv": 0.001,
    "lam_semi": 0.3,
    "t_semi": 0.3,
    "max_epoch": 80,
    "snapshot": '/model1.pth.tar',
    "snapshot_dir": '',
    "batch_size": 12,
    "val_orig": True,
    "d_label_smooth": 0.2, 
    "d_lr": 0.00001,
    "g_lr": 0.00025,
    "seed": 1, 
}

def train_semi(generator, discriminator, optimG, optimD, schedG, schedD, trainloader_l, trainloader_u, valoader, config):
    best_miou = -1
    criterion = nn.NLLLoss(ignore_index=255)
    epoch_LDr_losses = []
    epoch_LDf_losses = []
    epoch_LGce_losses = []
    epoch_LGadv_losses = []
    epoch_LGsemi_losses = []
    best_epoch = 0

    labeled_iterations_count = 0

    train_losses, val_losses = [], []
    val_miou = []

    torch.cuda.empty_cache()

    for e in range(40):
        generator.train()
        train_loss = 0
        for i, data in enumerate(tqdm(trainloader_l)):
            image, mask, _ = data
            image = image.cuda()
            mask = mask.cuda()

            cpmap = generator(image)
            output = torch.nn.functional.log_softmax(cpmap, dim=1)
            loss = criterion(output, mask)

            loss.backward()
            optimG.step() 
            optimG.zero_grad() 
            train_loss += loss.item()

        schedG.step()

        generator.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valoader)):
                image, mask, _ = data
                image = image.cuda()
                mask = mask.cuda()
                cpmap = generator(image)
                output = torch.nn.functional.log_softmax(cpmap, dim=1)
                loss = criterion(output, mask)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(trainloader_l))
        val_losses.append(val_loss / len(valoader))
        val_miou.append(val(generator, valoader))
        

        print("Epoch: {}/{} |".format(e+1, 40),
              "Train Loss: {:.3f} |".format(train_losses[-1]),
              "Val Loss: {:.3f} |".format(val_losses[-1]),
              "Val mIoU: {:.3f}".format(val_miou[-1]))

    for epoch in range(41, config['max_epoch'] + 1):
        generator.train()
        discriminator.train()
        trainloader_l_iter = iter(trainloader_l)
        trainloader_u_iter = iter(trainloader_u)
        print("Epoch: {}".format(epoch))
        batch_id = 0

        current_epoch_LDr = []
        current_epoch_LDf = []
        current_epoch_LGce = []
        current_epoch_LGadv = []
        current_epoch_LGsemi = []

        current_batch_gradG_norms = [] 
        current_batch_gradD_norms = [] 

        while True:
            if random.random() < 0.2:
                loader = trainloader_l_iter
                labeled = True
            else:
                loader = trainloader_u_iter
                labeled = False

            try:
                img, mask, ohmask = next(loader)
            except StopIteration:
                loader = trainloader_u_iter if labeled else trainloader_l_iter
                labeled = not labeled
                try:
                    img, mask, ohmask = next(loader)
                except StopIteration:
                    break

            batch_id += 1

            img = img.cuda()
            mask = mask.cuda()
            ohmask = ohmask.cuda()

            if labeled:
                cpmap = generator(img)
                cpmap = torch.nn.functional.softmax(cpmap, dim=1)

                N, _, H, W = cpmap.size()
                targetf = torch.zeros((N, H, W), dtype=torch.long, device=img.device)
                targetr = torch.ones((N, H, W), dtype=torch.long, device=img.device)
                optimD.zero_grad()

                # Train on Real
                confr = torch.nn.functional.log_softmax(discriminator(ohmask.float()), dim=1)

                if config['d_label_smooth'] != 0:
                    LDr = (1 - config['d_label_smooth']) * criterion(confr, targetr) + \
                          config['d_label_smooth'] * criterion(confr, targetf)
                else:
                    LDr = criterion(confr, targetr)
                LDr.backward()

                # Train on Fake
                conff = F.log_softmax(discriminator(cpmap), dim=1)
                LDf = criterion(conff, targetf)
                LDf.backward()

                current_epoch_LDr.append(LDr.item())
                current_epoch_LDf.append(LDf.item())
                
                # --- Monitor Discriminator Gradients (after both backward calls) ---
                d_grad_norm = 0
                for p in discriminator.parameters():
                    if p.grad is not None:
                        d_grad_norm += p.grad.norm(2).item() ** 2
                d_grad_norm = d_grad_norm ** 0.5
                current_batch_gradD_norms.append(d_grad_norm)
                
                optimD.step()
                schedD.step()

                #  labelled data Generator Training #
                optimG.zero_grad()
                optimD.zero_grad()

                cpmap = generator(img)
                cpmaplsmax = torch.nn.functional.log_softmax(cpmap, dim=1)

                conff = F.log_softmax(discriminator(cpmap), dim=1)
                LGce = criterion(cpmaplsmax, mask)
                
                # --- Randomly flip label for Generator's adversarial loss ---
                target_for_G_adv = torch.full((N, H, W), 0, dtype=torch.long, device=img.device) 
                if random.random() < 0.15: 
                    target_for_G_adv = torch.full((N, H, W), 1, dtype=torch.long, device=img.device)
                
                LGadv = criterion(conff, target_for_G_adv)

                current_epoch_LGce.append(LGce.item())
                current_epoch_LGadv.append(LGadv.item())
                current_epoch_LGsemi.append(0)

                LGadv_scaled = config['lam_adv'] * LGadv
                (LGce + LGadv_scaled).backward()
                
                g_grad_norm = 0
                for p in generator.parameters():
                    if p.grad is not None:
                        g_grad_norm += p.grad.norm(2).item() ** 2
                g_grad_norm = g_grad_norm ** 0.5
                current_batch_gradG_norms.append(g_grad_norm)
                
                optimG.step()
                schedG.step()

            else:
                current_epoch_LDr.append(0)
                current_epoch_LDf.append(0)

                optimG.zero_grad()
                if labeled_iterations_count >= 1000: 
                    cpmap = generator(img)
                    cpmapsmax = torch.nn.functional.softmax(cpmap, dim=1)
                    conf = discriminator(cpmap)
                    confsmax = torch.nn.functional.softmax(conf, dim=1)
                    conflsmax = torch.nn.functional.log_softmax(conf, dim=1)

                    N, _, H, W = cpmap.size()
                    target_for_G_adv_unlabeled = torch.full((N, H, W), 0, dtype=torch.long, device=img.device) 
                    if random.random() < 0.15: 
                        target_for_G_adv_unlabeled = torch.full((N, H, W), 1, dtype=torch.long, device=img.device)

                    LGadv = criterion(conflsmax, target_for_G_adv_unlabeled)
                    current_epoch_LGadv.append(LGadv.item())
                    current_epoch_LGce.append(0)

                    hardpred = torch.argmax(cpmapsmax, dim=1)
                    confnp = confsmax[:, 1, :, :].detach().cpu().numpy()
                    hardprednp = hardpred.detach().cpu().numpy()

                    pseudo_label = torch.full_like(hardpred, 255, dtype=torch.long)
                    confident_pixels_mask = torch.from_numpy(confnp > config['t_semi']).bool().to(img.device)
                    pseudo_label[confident_pixels_mask] = hardpred[confident_pixels_mask]

                    # Only calculate LGsemi if there are confident pixels
                    if torch.any(confident_pixels_mask):
                        LGsemi = F.cross_entropy(cpmap, pseudo_label, ignore_index=255)
                        LGsemi_d = LGsemi.item()
                        current_epoch_LGsemi.append(LGsemi_d)
                        LG = config['lam_adv'] * LGadv + config['lam_semi'] * LGsemi
                    else:
                        LG = config['lam_adv'] * LGadv
                        current_epoch_LGsemi.append(0)

                    LG.backward()
                    optimG.step()
                    schedG.step()

                    del confnp, confsmax, hardpred, hardprednp, cpmapsmax, cpmap
                else:
                    # If not enough labeled iterations, these losses are zero for unlabeled data
                    current_epoch_LGce.append(0)
                    current_epoch_LGadv.append(0)
                    current_epoch_LGsemi.append(0)
            # --- End of Batch ---

        # Calculate average losses for the epoch
        epoch_LDr_losses.append(np.mean(current_epoch_LDr) if current_epoch_LDr else 0)
        epoch_LDf_losses.append(np.mean(current_epoch_LDf) if current_epoch_LDf else 0)
        epoch_LGce_losses.append(np.mean(current_epoch_LGce) if current_epoch_LGce else 0)
        epoch_LGadv_losses.append(np.mean(current_epoch_LGadv) if current_epoch_LGadv else 0)
        epoch_LGsemi_losses.append(np.mean(current_epoch_LGsemi) if current_epoch_LGsemi else 0)

        # Print average losses for the current epoch
        print(f"Epoch {epoch} Losses: LDr_avg={epoch_LDr_losses[-1]:.4f}, LDf_avg={epoch_LDf_losses[-1]:.4f}, "
              f"LGce_avg={epoch_LGce_losses[-1]:.4f}, LGadv_avg={epoch_LGadv_losses[-1]:.4f}, LGsemi_avg={epoch_LGsemi_losses[-1]:.4f}")


        miou = val(generator, valoader)
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            torch.save(generator,os.path.join(config['snapshot_dir'],'{}.pt'.format(config['prefix'])))

        print("[{}] Curr mIoU: {:0.10f} Best mIoU: {}".format(epoch,miou,best_miou))
    print("Saved best snapshot from ",best_epoch, " epoch")
    return epoch_LDr_losses, epoch_LDf_losses, epoch_LGce_losses, epoch_LGadv_losses, epoch_LGsemi_losses

def main(home_dir):

    trainloader_l, trainloader_u, valoader = make_datasets(home_dir, config)

    generator = smp.DeepLabV3Plus('efficientnet-b2', encoder_weights='imagenet', 
                            classes=21, activation=None, encoder_depth=5, decoder_channels=256)
    generator.cuda()
    encoder_params = generator.encoder.parameters()
    decoder_params = list(generator.decoder.parameters()) + list(generator.segmentation_head.parameters())
    print(sum(p.numel() for p in generator.parameters() if p.requires_grad))
    optimG = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-4},
        {'params': decoder_params, 'lr': 1e-3}
    ], weight_decay=1e-4)

    discriminator = Dis(in_channels=21)

    optimD = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),lr = config['d_lr'],weight_decay=0.0001)
    discriminator.cuda()
    num_batches_per_epoch = len(trainloader_l) + len(trainloader_u)

    total_training_iterations = config['max_epoch'] * num_batches_per_epoch

    schedG = torch.optim.lr_scheduler.PolynomialLR(optimG, total_iters=total_training_iterations, power=0.9, last_epoch=-1)
    schedD = torch.optim.lr_scheduler.PolynomialLR(optimD, total_iters=total_training_iterations, power=0.9, last_epoch=-1)
    print("Semi-Supervised training")
    train_semi(generator,discriminator,optimG,optimD,schedG,schedD,trainloader_l,trainloader_u,valoader,config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("home_dir", help="enter the path to the folder with the dataset files",
                        type=str)
    args = parser.parse_args()
    main(args.home_dir)





