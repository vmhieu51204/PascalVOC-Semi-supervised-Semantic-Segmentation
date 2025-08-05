import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (320, 320)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 21
BATCH_SIZE = 12

#Helper functions
class OneHotEncode(object):
    """
        Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    """
    def __init__(self,nclass=21):
        self.nclass = nclass

    def __call__(self,label):
        label_a = np.array(transforms.ToPILImage()(label.byte().unsqueeze(0)),np.uint8)

        ohlabel = np.zeros((self.nclass,label_a.shape[0],label_a.shape[1])).astype(np.uint8)

        for c in range(self.nclass):
            ohlabel[c, :, :] = (label_a == c).astype(np.uint8)

        return torch.from_numpy(ohlabel)

def read_img_list(filename):
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line.strip())
    return np.array(img_list)

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def pascal_palette():
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

  return palette

def pascal_palette_invert():
  palette_list = pascal_palette().keys()
  palette = ()

  for color in palette_list:
    palette += color

  return palette

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=20000, power=0.9):
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - float(iter)/max_iter)**power
    return optimizer

#helper 2

def val(model, valoader, nclass=21, nogpu=False):
    model.eval()
    gts, preds = [], []

    with torch.no_grad():
        for img_id, (img, gt_mask, _) in enumerate(valoader):
            if not nogpu:
                img = img.cuda()
                out_pred_map = model(img).cpu().numpy()  # shape: (B, C, H, W)
            else:
                out_pred_map = model(img).numpy()

            gt_mask = gt_mask.numpy()  # shape: (B, H, W)

            for i in range(out_pred_map.shape[0]):
                soft_pred = out_pred_map[i]  # shape: (C, H, W)
                hard_pred = np.argmax(soft_pred, axis=0).astype(np.uint8)  # shape: (H, W)
                gt = gt_mask[i]  # shape: (H, W)

                # Ensure shapes match
                assert hard_pred.shape == gt.shape, f"Shape mismatch: pred {hard_pred.shape}, gt {gt.shape}"

                gts.append(gt)
                preds.append(hard_pred)

    miou, _ = scores(gts, preds, n_class=nclass)
    return miou

def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu)) #dictionary of per-class oui

    return mean_iu, cls_iu

def enable_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            for param in m.parameters():
                param.requires_grad = True

def val_mt_fused(generator_student, mt_network, valoader, nclass=21, fusion_threshold=0.2, device="cuda"):

    generator_student.eval() # Set student model to evaluation mode
    mt_network.teacher.eval() # Set teacher model to evaluation mode

    gts, preds = [], []

    with torch.no_grad(): # Disable gradient calculations for efficiency
        for img_id, (img, gt_mask, _) in enumerate(valoader): # _ is for one-hot mask, not used here
            img = img.to(device)

            student_logits = generator_student(img) # shape: (B, C, H, W)

            teacher_logits_from_mt, _ = mt_network(img) # MeanTeacherNetwork returns (student_out, teacher_out) in eval mode
            teacher_probs = F.softmax(teacher_logits_from_mt, dim=1) # Apply softmax to get probabilities

            deactivation_mask = (teacher_probs <= fusion_threshold)

            fused_logits = student_logits.clone() # Create a copy to modify
            fused_logits[deactivation_mask] = -1e9 # A sufficiently small number

            hard_pred = torch.argmax(fused_logits, dim=1).cpu().numpy() 
            gt_mask = gt_mask.numpy() # shape: (B, H, W)

            for i in range(hard_pred.shape[0]): # Iterate over batch
                pred_i = hard_pred[i]
                gt_i = gt_mask[i]
                assert pred_i.shape == gt_i.shape, f"Shape mismatch: pred {pred_i.shape}, gt {gt_i.shape}"

                gts.append(gt_i)
                preds.append(pred_i)
    miou, _ = scores(gts, preds, nclass)
    return miou

def denormalize_image_tensor(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation.
    Args:
        tensor (torch.Tensor): Tensor image of size (C, H, W) to be denormalized.
        mean (list): Mean for each channel.
        std (list): Standard deviation for each channel.
    Returns:
        torch.Tensor: Denormalized tensor image.
    """
    denormalized_tensor = tensor.clone().cpu()
    for t, m, s in zip(denormalized_tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(denormalized_tensor, 0, 1)

# Helper function to convert a segmentation mask tensor to an RGB PIL Image
def tensor_mask_to_rgb_pil(mask_tensor, flat_palette):
    """Converts a HxW tensor mask to an RGB PIL Image using a flat palette."""
    mask_numpy = mask_tensor.cpu().numpy().astype(np.uint8)
    mask_numpy[mask_numpy == 255] = 0 
    pil_image = Image.fromarray(mask_numpy, mode='P')
    pil_image.putpalette(flat_palette)
    return pil_image.convert('RGB')

# Helper function to get prediction from a standard segmentation model for a single image
def get_model_prediction_single_image(model, image_tensor_batch, device=DEVICE):
    """Generates prediction for a single image batch from a standard model."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor_batch)
        if isinstance(logits, tuple): # Handle models returning multiple outputs
            logits = logits[0]
        hard_pred = torch.argmax(logits.squeeze(0), dim=0) # (H, W)
        return hard_pred.cpu()

def get_mt_teacher_prediction_single_image(mt_network, image_tensor_batch, device=DEVICE):
    """Generates prediction for a single image batch from the MT's teacher model."""
    mt_network.eval()
    with torch.no_grad():
        _, teacher_logits = mt_network(image_tensor_batch)
        hard_pred = torch.argmax(teacher_logits.squeeze(0), dim=0) # (H, W)
        return hard_pred.cpu()

# Helper function for the special s4gan fused prediction (GAN student + MT student)
def get_s4gan_fused_prediction_single_image(gan_student_model, mt_network, image_tensor_batch, fusion_threshold=0.2, device=DEVICE):
    """Generates fused prediction for a single image batch (GAN student + MT student)."""
    gan_student_model.eval()
    mt_network.eval() 

    with torch.no_grad():
        gan_student_logits = gan_student_model(image_tensor_batch)
        _ , mt_student_logits = mt_network(image_tensor_batch)
        mt_student_probs = F.softmax(mt_student_logits, dim=1)
        deactivation_mask = (mt_student_probs <= fusion_threshold)
        fused_logits = gan_student_logits.clone()
        fused_logits[deactivation_mask] = -1e9
        hard_pred = torch.argmax(fused_logits.squeeze(0), dim=0)
        return hard_pred.cpu()


class mIoU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return np.nanmean(iu)

def pixel_accuracy(output, mask, ignore_index=255):
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        valid = (mask != ignore_index)
        correct = torch.eq(output, mask).int() & valid
        accuracy = float(correct.sum()) / float(valid.sum())
    return accuracy