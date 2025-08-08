# PascalVOC-Semi-supervised-Semantic-Segmentation
Semantic segmentation is a fundamental task in computer vision, but state-of-the-art models typically demand large-scale, pixel-level annotated datasets, which are costly and labor-intensive to produce. This project
investigates the efficacy of semi-supervised learning (SSL) as a paradigm to mitigate this dependency by training models on a small set of labeled images and a large set of unlabeled images. The project implement and evaluate GAN and S4GAN model, built upon a DeepLabV3+ architecture with an EfficientNet-B2 encoder. Experiments conducted on the PASCAL VOC 2012 and Segmentation Boundary Dataset (SBD) demonstrate that most semisupervised approaches significantly outperform a baseline model trained only on the limited labeled data.

![alt text](image.png)

## Results
All experiments were conducted on the Kaggle Notebook platform, utilizing an NVIDIA Tesla
P100 GPU with 16 GB of VRAM. The project setup uses a supervised-to-unsupervised data ratio of
approximately 1/5. 

The implemented methods is evaluated against a supervised-only baseline and a fully supervised model. The former was trained only on the small labeled set (1,464 images), 
while the latter was trained on the full labeled dataset (VOC Train + SBD, 1,464 + 5,623 images)
### Table: Performance comparison of various models under different training configurations

| **Train Dataset**            | **Model**      | **Mean PA** | **Mean IoU** | **% Δ Mean PA** | **% Δ Mean IoU** |
|-----------------------------|----------------|-------------|--------------|------------------|-------------------|
| VOC Labeled                 | Base model     | 0.9361      | 0.7243       | 0.00%            | 0.00%             |
| VOC Labeled + SBD Labeled   | Full model     | 0.9446      | 0.7590       | +0.91%           | +4.79%            |
| VOC Labeled + SBD Unlabeled | Mean Teacher   | 0.9413      | 0.7464       | +0.56%           | +3.06%            |
| VOC Labeled + SBD Unlabeled | GAN            | 0.9348      | 0.7235       | -0.14%           | -0.11%            |
| VOC Labeled + SBD Unlabeled | s4GAN          | **0.9415**  | **0.7473**   | **+0.58%**       | **+3.18%**        |

From the results, we can see that most semi-supervised
methods significantly outperform the baseline model. This validates the core hypothesis that leveraging unlabeled data can effectively bridge
the performance gap when labeled data is scarce.

The standard GAN approach (0.7235 mIoU) slightly underperformed the baseline,
suggesting that a naive adversarial setup can be unstable without the more advanced mechanisms
present in S4GAN.


