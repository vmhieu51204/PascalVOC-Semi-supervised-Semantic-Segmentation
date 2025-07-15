from torchvision.datasets import SBDataset
from torch.utils.data import Dataset, DataLoader
import os
import random
from utils.utils import *
from PIL import Image
from augmentations.augmentations import get_albu_transform, get_val_transform, AlbuTransform

IMAGE_SIZE = (320, 320)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

#/kaggle/input/pascal-voc-2012/
class PascalVOC(Dataset):
    def __init__(self, root,indices=None, co_transform=None, train_phase=True):
        self.n_class = 21
        self.root = root
        self.images_root = os.path.join(self.root, 'JPEGImages')
        self.labels_root = os.path.join(self.root, 'SegmentationClass')
        self.img_list = read_img_list(os.path.join(root,'ImageSets/Segmentation/train.txt')) \
                            if train_phase else read_img_list(os.path.join(self.root,'ImageSets/Segmentation/val.txt'))
        if indices is not None:
            print('')
        self.co_transform = co_transform
        self.train_phase = train_phase

    def __getitem__(self, index):
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.jpg'), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(os.path.join(self.labels_root,filename+'.png'), 'rb') as f:
            label = Image.open(f).convert('P')

        image, label = self.co_transform(image, label)
        label = torch.as_tensor(label, dtype=torch.long)
        ohlabel = OneHotEncode()(label)

        return image, label, ohlabel
        
    def __len__(self):
        return len(self.img_list)

class VOCPseudoLabel(Dataset):
    def __init__(self, base_set, transform=None):
        self.base_set = base_set
        self.transform = transform

    def __len__(self):
        return len(self.base_set)

    def __getitem__(self, idx):
        image, _ = self.base_set[idx]
        image = np.array(image)
        image = A.Resize(*(320, 320))(image=image)["image"]
        image = self.transform(image)
        dummy_mask = torch.empty(IMAGE_SIZE, dtype=torch.long).fill_(255) 
        dummy_ohmask = torch.empty(21, *IMAGE_SIZE, dtype=torch.uint8) 

        return image, dummy_mask, dummy_ohmask

def make_datasets(home_dir, config):
    download_sb = not os.path.isdir("sb_dataset")
    sbd_train = SBDataset("sb_dataset", image_set="train_noval", mode="segmentation", download=download_sb)

    labes_root = os.path.join('/kaggle/input/pascal-voc-2012-dataset/VOC2012_train_val/VOC2012_train_val/', 'SegmentationClass')
    os.makedirs("pseudo_label", exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    all_img_list = read_img_list(os.path.join(home_dir, 'ImageSets/Segmentation/train.txt'))
    n_total = len(all_img_list)
    np.random.seed(config['seed'])
    idx_labeled = np.random.choice(n_total, int(n_total * config['split']), replace=False)
    idx_unlabeled = np.setdiff1d(np.arange(n_total), idx_labeled)

    co_transform = AlbuTransform(get_albu_transform())

    trainset_l = PascalVOC(
        root=home_dir,
        indices=idx_labeled,
        co_transform=co_transform,
        train_phase=True,
    )

    trainset_u = VOCPseudoLabel(sbd_train, transform=co_transform)
    trainloader_l = DataLoader(trainset_l,batch_size=config['batch_size'], shuffle=True,num_workers=2,drop_last=True)
    trainloader_u = DataLoader(trainset_u,batch_size=config['batch_size'], shuffle=True,num_workers=2,drop_last=True)
    co_transform_val = AlbuTransform(get_val_transform())

    valset = PascalVOC(home_dir,co_transform=co_transform_val,train_phase=False)
    valoader = DataLoader(valset,batch_size=config['batch_size'], shuffle=False, num_workers=2,drop_last=False)