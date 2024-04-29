import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import cv2
import random




class GlaS2015(Dataset):
    def __init__(self, path=None, original:bool=True, split='train', exp_name_cp=None, exp_name_gan=None, imsize=512, for_segmentation=False):
        assert split in ['train', 'test']
        self.split = split
        self.path = path
        self.exp_name_cp = exp_name_cp
        self.exp_name_gan = exp_name_gan
        self.imsize = int(imsize)
        self.for_segmentation = for_segmentation
        self.df = self.get_sample_info(original)
        
    
    def get_sample_info(self, original):
        image_list = []
        mask_list = []
        grade_list = []
        if original:
            annotation = pd.read_csv(fr'{self.path}/data/Warwick QU Dataset (Released 2016_07_08)/Grade.csv')
            annotation = annotation[['name', ' grade (GlaS)']]
            for idx, row in annotation.iterrows():
                name = row['name']
                grade = {' benign':0, ' malignant':1}[row[' grade (GlaS)']]
                if self.split not in name:
                    continue
                image_list.append(f'Warwick QU Dataset (Released 2016_07_08)/{name}.bmp')
                mask_list.append(f'Warwick QU Dataset (Released 2016_07_08)/{name}_anno.bmp')
                grade_list.append(grade)
        if self.exp_name_cp is not None:
            annotation = pd.read_csv(fr'{self.path}/data/bank/cp/{self.exp_name_cp}/annotation.csv')
            split_idx = int(0.8*len(annotation))
            annotation = annotation[0:split_idx] if self.split=='train' else annotation[split_idx:]
            annotation.reset_index()
            for idx, row in annotation.iterrows():
                name = row['name']
                grade = row['grade']
                image_list.append(f'bank/cp/{self.exp_name_cp}/{name}')
                mask_list.append(f'bank/cp/{self.exp_name_cp}/{name.rstrip(".png")+"_mask.png"}')
                grade_list.append(grade)
        if self.exp_name_gan is not None:
            annotation = pd.read_csv(fr'{self.path}/data/bank/gan/{self.exp_name_gan}/samples/annotation.csv')
            split_idx = int(0.8*len(annotation))
            annotation = annotation[0:split_idx] if self.split=='train' else annotation[split_idx:]
            annotation.reset_index()
            for idx, row in annotation.iterrows():
                name = row['name']
                grade = row['grade']
                image_list.append(f'bank/gan/{self.exp_name_gan}/samples/{name}')
                mask_list.append(f'bank/gan/{self.exp_name_gan}/samples/{name.rstrip(".png")+"_mask.png"}')
                grade_list.append(grade)
        
        df = pd.DataFrame(data={'image':image_list, 'mask':mask_list, 'grade':grade_list})
        
        # return
        return df


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        # locations and grade info
        row = self.df.loc[idx]
        image_path, mask_path, grade = row['image'], row['mask'], row['grade']

        # load image, mask
        image = cv2.imread(fr'{self.path}/data/{image_path}', cv2.IMREAD_COLOR) # (h,w,3)
        mask = cv2.imread(fr'{self.path}/data/{mask_path}', cv2.IMREAD_GRAYSCALE) # (h,w)

        # resize
        image = cv2.resize(image, (self.imsize, self.imsize))
        mask = cv2.resize(mask, (self.imsize, self.imsize))

        # mask reduction
        mask = np.where(mask>=0.5, 1.0, 0)

        # mask expand
        mask = mask[None,...] # (1, h, 2)
        if self.for_segmentation:
            mask = np.stack([np.ones_like(mask)-mask, mask], axis=-1) # (2, h , w)

        # reshape to make channel-first-image
        image = np.einsum('...c->c...', image)

        # normalize
        image = image/255
        image = (image-0.5)/0.5

        # type
        image = torch.from_numpy(image).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        grade = torch.tensor(grade).type(torch.LongTensor)

        # return
        return image, mask, grade



if __name__ == '__main__':
    path = 'E:\sungrae\CaPAGAN'
    dataset = GlaS2015(path, True, 'test', 2, None, 512)
    dataloader = DataLoader(dataset, 8, shuffle=True)
    image, mask, grade = next(iter(dataloader))

    print(f'dataloader : {len(dataloader)}')
    print(f'image : {image.shape}, {type(image)}, [{torch.min(image).item()}, {torch.max(image).item()}]') # (b,3,imsize,imsize)
    print(f'mask : {mask.shape}, {type(mask)}, [{torch.min(mask).item()}, {torch.max(mask).item()}]') # (b,1,imsize,imsize)
    print(f'grade : {grade.shape}, {type(grade)}') # (b,1)