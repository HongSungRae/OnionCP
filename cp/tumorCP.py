import numpy as np
import random
from skimage.filters import gaussian
import albumentations as A
import cv2




def tumor_cp(img_source, img_target, mask_source, mask_target, p_cp=1, p_trans=0.5):
    '''
    > Implementation of TumorCP[Yang et al., MICAAI 2021]
    = input =
    img (size, size, 3) -> numpy.ndarray
    mask (size, size) -> numpy.ndarray
    p_cp : tumor_cp probability
    p_trans : probability for rigid augmentation

    = output =
    CP-Tumor augmented img and mask
    img (size, size, 3) -> numpy.ndarray
    mask (size, size) -> numpy.ndarray. It contains 0 or 1.
    '''
    size = img_source.shape[1]

    # step1 : Do TumorCP with P_cp
    if p_cp >= random.random():
        # step2 : Randomly select a tumor to copy
        img_source = img_source * mask_source[...,None] # image with tumor only # (size,size,3)

        # step3 : Do Object-level augmentation with P_trans
        ## step3-1 : rigid transformation
        transform_rigid = A.Compose([A.HorizontalFlip(p=p_trans),
                                     A.VerticalFlip(p=p_trans),
                                     A.RandomRotate90(p=p_trans),
                                     A.Rotate(limit=(-180, 180),p=p_trans), # =(-pi, pi)
                                     A.RandomScale(scale_limit=(-0.25,0.25),p=p_trans), # =(0.75,1.25), I scaled from paper's ratio to follow albumentation's range
                                     A.PadIfNeeded(size,size),
                                     A.Resize(size,size)
                                     ]) # a.k.a spatial transformation in the paper
        transformed = transform_rigid(image=img_source, mask=mask_source)
        img_source = transformed['image'] # (size,size,3) np.ndarray
        mask_source = transformed['mask'] # (size,size) np.ndarray

        ## step3-2 : gamma transformation
        transform_gamma = A.Compose([A.RandomGamma(gamma_limit=(75,150),p=1),
                                     A.Resize(size,size)]) # gamma, I scaled from paper's ratio to follow albumentation's range
        transformed = transform_gamma(image=img_source.astype(np.uint8), mask=mask_source) # A.RandomGamma uses Numpy's function. It requires float64 type.
        img_source = transformed['image'] # (size,size,3) np.ndarray
        mask_source = transformed['mask'] # (size,size,2) np.ndarray

        ## step3-3 : blurring(gaussian) transformation
        sigma = random.randint(50,100)/100
        mask_source = gaussian(mask_source, sigma=sigma) # (size,size)

        # step4 : Randomly select a place to paste onto
        img = img_source*mask_source[...,None] + img_target*(1-mask_source[...,None])
        mask = np.where((mask_target + mask_source)>0, 1, 0)
        
    else: # step1 : do nothing with prop (1-P_cp)
        img, mask = img_target, mask_target
    

    # step5 : Image-level Data Augmentation
    transform_img_level = A.Compose([A.Rotate(limit=(-30,30),p=0.2),
                                     A.RandomScale(scale_limit=(0.7,1.4),p=0.2),
                                     A.GaussNoise(p=0.15),
                                     A.GaussianBlur(p=0.2),
                                     A.RandomBrightness(limit=(-.2,.2), p=0.15),
                                     A.RandomContrast(limit=(-.2,.2), p=0.15),
                                     A.Compose([A.Resize(int(size/2), int(size/2), interpolation=cv2.INTER_NEAREST),
                                                A.Resize(size,size, interpolation=cv2.INTER_CUBIC)],
                                                p=0.25),
                                     A.RandomGamma(p=0.15),
                                     A.HorizontalFlip(),
                                     A.VerticalFlip(),
                                     A.Resize(size,size)])
    transformed = transform_img_level(image=img.astype(np.uint8), mask=mask)
    img = transformed['image']
    mask = transformed['mask']

    return img, mask