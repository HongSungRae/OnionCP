# library
import cv2
import numpy as np
from skimage import measure
import random



def inpainting_cp(image_source, image_target, mask_source, mask_target):
    '''
    Copy-and-Paste source and target.
    Please refer details from paper.

    = input =
    image_* : np.ndarray. RGB [0, 255]. (h, w, 3) shape
    mask_* : np.ndarray. [0 or 1]. (h, w) shape

    = return =
    image_cp : np.ndarray. RGB [0, 255]. (imsize, imsize, 3) shape
    mask_cp : np.ndarray. [0 or 1]. (imsize, imsize) shape
    '''
    # image random rotation $ flip
    image_source, mask_source = augment(image_source, mask_source)
    image_target, mask_target = augment(image_target, mask_target)

    # image to Lab space
    image_source = cv2.cvtColor(image_source, cv2.COLOR_RGB2Lab)
    image_target = cv2.cvtColor(image_target, cv2.COLOR_RGB2Lab)

    # target image object erasing and inpainting
    inpatined_target = cv2.inpaint(image_target, mask_target[...,None].astype(np.uint8), 3, cv2.INPAINT_TELEA)

    # source image warping
    image_source_warped, mask_source_warped = warp(image_source, mask_source)

    # normalize and shift
    object_source = image_source_warped * np.where(mask_source_warped[...,None]>=1.0, 1.0, 0)
    object_only_source = image_source_warped[np.where(mask_source_warped>=1.0, 1.0, 0)==1]
    object_only_target = image_target[np.where(mask_target>=1.0, 1.0, 0)==1]

    object_source_R, object_source_G, object_source_B = object_source[...,0], object_source[...,1], object_source[...,2]
    s_R, s_G, s_B = object_only_source[...,0], object_only_source[...,1], object_only_source[...,2]
    t_R, t_G, t_B = object_only_target[...,0], object_only_target[...,1], object_only_target[...,2]

    object_source_R = np.std(t_R) * ((object_source_R - np.mean(s_R))/np.std(s_R)) + np.mean(t_R)
    object_source_G = np.std(t_G) * ((object_source_G - np.mean(s_G))/np.std(s_G)) + np.mean(t_G)
    object_source_B = np.std(t_B) * ((object_source_B - np.mean(s_B))/np.std(s_B)) + np.mean(t_B)

    object_source = np.stack((object_source_R, object_source_G, object_source_B), axis=-1)
    object_source = np.clip(object_source, 0, 255)

    # select randomly U(1, n(O_s)) object(s)
    unique_objects = measure.label(mask_source_warped, connectivity=2)
    num_objects = unique_objects.max()
    n = 1 if num_objects==1 else random.randint(1, num_objects)
    random_uniform_idx = random.sample(range(1, num_objects+1), n)
    mask_cp = np.zeros_like(mask_source_warped)
    for idx in random_uniform_idx:
        mask_cp += (unique_objects==idx)
    
    # Copy-and-Paste : erase object from target and paste object from source
    image_cp = inpatined_target * (1-mask_cp[...,None]) + object_source * mask_cp[...,None]

    # type cast
    image_cp, mask_cp = image_cp.astype(np.uint8), mask_cp.astype(np.uint8)
    image_cp = cv2.cvtColor(image_cp, cv2.COLOR_Lab2RGB)

    # return
    return image_cp, mask_cp



def augment(image, mask):
    '''
    Spatial Augmentations
    = input =
    image : np.ndarray. RGB [0, 255]. (h, w, 3) shape
    mask : np.ndarray. [0 or 1]. (h, w) shape

    = return = 
    image : np.ndarray. RGB [0, 255]. (h, w, 3) shape
    mask : np.ndarray. [0 or 1]. (h, w) shape
    '''
    # rotation
    rotations_90 = random.choice([0,1,2,3])
    for _ in range(rotations_90):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    # flip
    hor = random.choice([True, False])
    ver = random.choice([True, False])
    if hor:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if ver:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask



def warp(image, mask):
    '''
    The function warps given images and mask with r=16 of 2D Guassian circle.
    If imsize = 512, there are 16*16=256 circles. (512/(2*r))**2 = 256.

    = input =
    image : np.ndarray. [0, 255]. (h, w, 3) shape
    mask : np.ndarray. [0 or 1]. (h, w) shape

    = return =
    image_warped : np.ndarray. [0, 255]. (h, w, 3) shape
    mask_warped : np.ndarray. [0 or 1]. (h, w) shape
    '''
    # init information
    height, width = image.shape[0:2]

    # get warping coordinates
    _from, _to = get_warp_points(height)

    # warping transformations
    transformation_matrix, _ = cv2.findHomography(_from, _to)
    image_warped = cv2.warpPerspective(image, transformation_matrix, (height, width))
    mask_warped = cv2.warpPerspective(mask, transformation_matrix, (height, width))

    # return
    return image_warped, mask_warped



def get_warp_points(imsize):
    '''
    = input =
    imsize = image size(expected width==height)

    = return =
    _from : circle centeroid coordinates : np.array
    _to : Gaussian moved point coordinates : np.array
    '''
    # init variables
    _from = []
    _to = []

    grid_size = 64
    num_points = int((imsize//grid_size*2)**2)
    step = imsize//grid_size
    half_step = step//2
    count = 0

    # make _from points. i.e. centroid of Gaussian
    for y in range(half_step, imsize, step):
        for x in range(half_step, imsize, step):
            if count < num_points:
                _from.append([x, y])
                count += 1

    # make destination _to
    for centroid in _from:
        x, y = centroid
        std_dev = (grid_size**2)//2
        cov = std_dev * np.eye(2)
        x_prime, y_prime = np.random.multivariate_normal([x,y], cov)
        x_prime = max(min(x+15, x_prime),x-15)
        y_prime = max(min(y+15, y_prime),y-15)
        _to.append([int(x_prime), int(y_prime)])

    # type cast
    _from = np.array(_from, dtype=np.float32)
    _to = np.array(_to, dtype=np.float32)

    # return
    return _from, _to