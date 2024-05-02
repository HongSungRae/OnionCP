import cv2
import random
import os
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import tifffile
import xml.etree.ElementTree as ET


# Segmentation Dataset for MoNuSeg2018
class MoNuSeg2018(Dataset):
    def __init__(self, dataset_path=None, imsize=512, split=None, gen_augmentation=None, conven_augmentation=False, p=(0.5,0.5)):
        assert isinstance(imsize, int), 'Type Error'
        assert split in ['train', 'test'], 'Split Error'
        assert gen_augmentation in [None, 'cp', 'cpSimple', 'inpaintingCP', 'tumorCP', 'onionCP', 'gan', 'ddpm'], 'Generative augmentation Error'
        assert os.path.exists(dataset_path), "Dataset path Error."
        assert len(p) == 2

        self.dataset_path = dataset_path
        self.imsize = imsize
        self.split = split
        self.gen_augmentation = gen_augmentation
        self.conven_augmentation = conven_augmentation
        self.p = p
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parents_path = os.path.abspath(os.path.join(current_dir, ".."))

       
        self.data_list = os.listdir(fr'{dataset_path}/{split}/Tissue Images')

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
            image = cv2.imread(fr'{self.parents_path}/synthesized/monuseg/{self.gen_augmentation}/{i}.png', cv2.IMREAD_COLOR)
            mask = cv2.imread(fr'{self.parents_path}/synthesized/monuseg/{self.gen_augmentation}/{i}_mask.png', cv2.IMREAD_GRAYSCALE)
        else:
            image = tifffile.imread(fr'{self.dataset_path}/{self.split}/Tissue Images/{self.data_list[idx]}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                mask = cv2.imdecode(np.fromfile(fr'{self.dataset_path}/{self.split}/Annotations/{self.data_list[idx]}', np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError()
            except:
                imsize = 1000
                tree = ET.parse(fr'{self.dataset_path}/{self.split}/Annotations/{self.data_list[idx].split(".")[0]}.xml')
                root = tree.getroot()
                mask = np.zeros((imsize, imsize), dtype=np.uint8)
                color = 1
                for region in root.findall('.//Region'):
                    vertices = region.findall('.//Vertex')
                    points = [(float(vertex.attrib['X']), float(vertex.attrib['Y'])) for vertex in vertices]

                    # make ploy
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], color)
                    color += 1
                cv2.imwrite(fr'{self.dataset_path}/{self.split}/Annotations/{self.data_list[idx]}', mask)

        # resize
        image = cv2.resize(image.astype(np.uint8), (self.imsize, self.imsize))
        mask = np.where(cv2.resize(mask.astype(np.uint8), (self.imsize, self.imsize))>0, 1, 0)

        # conventional augmentations
        if self.conven_augmentation and likeli_conve <= self.p[1]:
            augmented = self.aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return  np.einsum('...c -> c...', image)/255, mask



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = MoNuSeg2018(fr'E:/MoNuSeg2018', 512, 'train', None, False)
    dataloader = DataLoader(dataset, 8, True)
    image, mask = next(iter(dataloader))
    print(f'Image : {image.shape}, Mask : {mask.shape}')