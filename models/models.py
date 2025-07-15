import torch.nn as nn
import torch
import segmentation_models_pytorch as smp



IMAGE_SIZE = (320, 320)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class Dis(nn.Module):
    def __init__(self, in_channels, negative_slope=0.2):
        super(Dis, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=2)
        self.conv5 = nn.Conv2d(512, 2, kernel_size=4, stride=2, padding=2)

        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        orig_size = x.shape[2:]
        x = self.lrelu(self.conv1(x))  # -> (N, 64, H/2, W/2)
        x = self.lrelu(self.conv2(x))  # -> (N, 128, H/4, W/4)
        x = self.lrelu(self.conv3(x))  # -> (N, 256, H/8, W/8)
        x = self.lrelu(self.conv4(x))  # -> (N, 512, H/16, W/16)
        x = self.conv5(x)              # -> (N, 2, H/32, W/32)

        for _ in range(5):  # 5x upsampling (x2^5 = 32)
            x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)

        return x
    
class MeanTeacherNetwork(nn.Module):
    def __init__(self, backbone, num_classes, ema_decay=0.999):
        super(MeanTeacherNetwork, self).__init__()
        self.ema_decay=ema_decay
        self.student = smp.DeepLabV3Plus(backbone, encoder_weights='imagenet', classes=num_classes, activation=None, encoder_depth=5, decoder_channels=256)
        self.teacher = smp.DeepLabV3Plus(backbone, encoder_weights=None, classes=num_classes, activation=None, encoder_depth=5, decoder_channels=256)

        for param in self.teacher.parameters():
            param.detach_()

    def forward(self, data, step=1, cur_iter=None):
        if not self.training:
            return self.student(data), self.teacher(data)
        # copy the parameters from teacher to student
        if cur_iter == 0:
            print("Copying parameters from student to teacher")
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.copy_(s_param.data)

        s_out = self.student(data)
        with torch.no_grad():
            t_out = self.teacher(data)

        if step == 1:
            self._update_ema_variables(self.ema_decay)

        return s_out, t_out

    def _update_ema_variables(self, ema_decay):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)