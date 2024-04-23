import torch
import albumentations as A
import cv2
import numpy as np
from skimage.filters import gaussian

# local
from cp import cp


def cp_simple(img_source, img_target, mask_source, mask_target, gaussian_blur=False, sigma=0.4):
    '''
    = input =
    img : (h, w, 3) shape of np.ndarry
    mask : img : (h, w) shape of np.ndarry. It contains [0 or 1].

    = output =
    Synthesized sample.
    img : img : (h, w, 3) shape of np.ndarry
    mask : img : (h, w) shape of np.ndarry. It contains [0 or 1].
    '''
    assert len(mask_source.shape) == 2
    assert len(mask_target.shape) == 2
    assert isinstance(img_source, np.ndarray)
    assert isinstance(img_target, np.ndarray)
    assert isinstance(mask_source, np.ndarray)
    assert isinstance(mask_target, np.ndarray)

    size = img_source.shape[1]
    transform = A.Compose([A.RandomScale(scale_limit=(-0.9, 1), p=1),
                           A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT),
                           A.HorizontalFlip(),
                           A.RandomCrop(size, size),
                           ])
    transformed = transform(image=img_source, mask=mask_source)
    img_source, mask_source = transformed['image'], transformed['mask']
    transformed = transform(image=img_target, mask=mask_target)
    img_target, mask_target = transformed['image'], transformed['mask']

    if gaussian_blur:
        '''
        Ghiasi et al., (the author of CP-simple) said
        *we also found that simply composing without any (gaussian) blending has similar performance*
        '''
        mask_source = gaussian(mask_source, sigma)

    return cp(img_source, img_target, mask_source, mask_target)