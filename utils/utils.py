import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn


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
