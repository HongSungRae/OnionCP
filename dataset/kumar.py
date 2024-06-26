import cv2
import random
from scipy import io
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
from dataset import dataset_path

# Segmentation Dataset for CPM17
class Kumar(Dataset):
    def __init__(self, dataset_path=dataset_path['kumar'], imsize=512, split='train', gen_augmentation=None, conven_augmentation=False, p=(0.5,0.5), diffusion=False, alpha_bar=None, T=None):
        assert isinstance(imsize, int), 'Type Error'
        assert split in ['train', 'test_same', 'test_diff'], 'Split Error'
        assert gen_augmentation in [None, 'cp', 'cpSimple', 'inpaintingCP', 'tumorCP', 'onionCP', 'gan', 'ddpm'], 'Generative augmentation Error'
        assert os.path.exists(dataset_path), "Dataset path Error."
        assert len(p) == 2

        self.dataset_path = dataset_path
        self.imsize = imsize
        self.split = split
        self.gen_augmentation = gen_augmentation
        self.conven_augmentation = conven_augmentation
        self.p = p
        self.diffusion = diffusion
        self.alpha_bar = alpha_bar
        self.T = T
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parents_path = os.path.abspath(os.path.join(current_dir, ".."))

        self.data_list = os.listdir(fr'{dataset_path}/{self.split}/Images')

        self.transform = A.Compose([A.Normalize(),
                                    ToTensorV2()])

        if conven_augmentation:
            self.aug = A.Compose([A.Rotate(limit=(-30,30),p=0.2),
                                  A.RandomScale(scale_limit=(0.7,1.4),p=0.2),
                                  A.GaussNoise(p=0.15),
                                  A.GaussianBlur(p=0.2),
                                  A.RandomBrightnessContrast(p=0.15),
                                  A.RandomGamma(p=0.15),
                                  A.HorizontalFlip(),
                                  A.VerticalFlip(),
                                  A.Resize(imsize,imsize)])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        likeli_gen = random.random() if self.split == 'train' else 99
        likeli_conve = random.random() if self.split == 'train' else 99

        # Load original or sythesized images
        if self.gen_augmentation is not None and likeli_gen <= self.p[0]:
            i = random.randint(1,5000)
            image = cv2.imread(fr'{self.parents_path}/synthesized/kumar/{self.gen_augmentation}/{i}.png', cv2.IMREAD_COLOR)
            mask = cv2.imread(fr'{self.parents_path}/synthesized/kumar/{self.gen_augmentation}/{i}_mask.png', cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(fr'{self.dataset_path}/{self.split}/Images/{self.data_list[idx]}', cv2.IMREAD_COLOR)
            mask = io.loadmat(fr'{self.dataset_path}/{self.split}/Labels/{self.data_list[idx].split(".")[0]}.mat')['inst_map']
            
        # resize
        image = cv2.resize(image, (self.imsize, self.imsize))
        mask = np.where(cv2.resize(mask.astype(np.uint8), (self.imsize, self.imsize))>0, 1, 0)

        # conventional augmentations
        if self.conven_augmentation and likeli_conve <= self.p[1]:
            augmented = self.aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # normalize and ToTensor
        augmented = self.transform(image=image)
        image = augmented['image']

        # diffusion data?
        if self.diffusion:
            t = np.random.randint(1, self.T+1)
            eps = torch.randn_like(image)
            x_t = torch.sqrt(self.alpha_bar[t]) * image + torch.sqrt(1 - self.alpha_bar[t]) * eps
            return {"img": image, "x_t": x_t, "t": t, "eps": eps}
        else:
            return  image, torch.from_numpy(mask).to(torch.float64)







if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    dataset = Kumar(fr'E:/kumar', 512, split='test_diff', gen_augmentation='cp', conven_augmentation=True)
    dataloader = DataLoader(dataset, 8, True)
    image, mask = next(iter(dataloader))
    print(f'Image : {image.shape}, Mask : {mask.shape}')