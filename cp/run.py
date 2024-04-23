# library
from tqdm import tqdm
from scipy import io
import random
import cv2
import numpy as np
import os
import sys
import time
import tifffile
import torch

# local
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(path)

from utils import misc, metric
from cp import cp
from cpSimple import cp_simple
from inpaintingCP import inpainting_cp
from tumorCP import tumor_cp
from onionCP import onion_cp


def run(cp_type=None, n_samples:int=10, dataset:str='', imsize:int=512, self_mix=False):
    '''
    = input =
    n_samples : the number of samples that user want to synthesize by CP
    dataset : dataset
    imsize : image size
    self_mix : if True image_source == image_target. i.e. self-oriented-mix
    
    = output =
    It saves (image_cp, mask_cp, annotation_df) locally.
    Save path is "./synthesized/dataset/cp_type".
    '''
    # assertion
    assert cp_type in ['cp', 'cpSimple', 'inpaintingCP', 'tumorCP', 'onionCP']
    assert n_samples >= 1
    assert dataset in ['glas2015', 'kumar', 'cpm17', 'crag']
    assert (imsize >= 256) and (imsize%(2**4) == 0)

    # make save path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, ".."))
    misc.make_dir(fr'{path}/synthesized/{dataset}/{cp_type}')

    # load data info
    dataset_path = misc.open_yaml(fr'{path}/configuration.yaml')[dataset]

    # synthesize new samples
    count = 1
    inception = None
    fid = None
    with tqdm(total=n_samples) as pbar:
        while count < n_samples+1:
            if dataset in ['glas2015', 'crag']:
                # load image and mask
                image_list = [item for item in os.listdir(fr'{dataset_path}') if "train" in item and "anno" not in item]
                if self_mix:
                    source_image_name = random.sample(image_list, 1)[0]
                    target_image_name = source_image_name
                else:
                    source_image_name, target_image_name = random.sample(image_list, 2)
                source_image = cv2.imread(fr'{dataset_path}/{source_image_name}', cv2.IMREAD_COLOR)
                source_mask = cv2.imread(fr'{dataset_path}/{source_image_name[0:-4] + "_anno.bmp"}', cv2.IMREAD_GRAYSCALE)
                target_image = cv2.imread(fr'{dataset_path}/{target_image_name}', cv2.IMREAD_COLOR)
                target_mask = cv2.imread(fr'{dataset_path}/{target_image_name[0:-4] + "_anno.bmp"}', cv2.IMREAD_GRAYSCALE)
            else:
                # load image and mask
                image_list = os.listdir(fr'{dataset_path}/train/Images')
                if self_mix:
                    source_image_name = random.sample(image_list, 1)[0]
                    target_image_name = source_image_name
                else:
                    source_image_name, target_image_name = random.sample(image_list, 2)

                if dataset == 'kumar':
                    source_image = tifffile.imread(fr'{dataset_path}/train/Images/{source_image_name}')
                    target_image = tifffile.imread(fr'{dataset_path}/train/Images/{target_image_name}')
                elif dataset == 'cpm17':
                    source_image = cv2.imread(fr'{dataset_path}/train/Images/{source_image_name}', cv2.IMREAD_COLOR)
                    target_image = cv2.imread(fr'{dataset_path}/train/Images/{target_image_name}', cv2.IMREAD_COLOR)
                source_mask = io.loadmat(fr'{dataset_path}/train/Labels/{source_image_name.split(".")[0]}.mat')['inst_map']
                target_mask = io.loadmat(fr'{dataset_path}/train/Labels/{target_image_name.split(".")[0]}.mat')['inst_map']
            
            if cp_type == 'onionCP':
                pass
            else:
                # select pasted objects
                num_objects = random.randint(1, np.max(source_mask))
                try:
                    selected_objects = np.array(random.sample(np.unique(source_mask).tolist()[1:], num_objects))
                except:
                    print(num_objects, np.unique(source_mask))
                source_mask = np.where(np.isin(source_mask,selected_objects), 1, 0)
                
                # resize
                source_image = cv2.resize(source_image, (imsize,imsize))
                source_mask = np.where(cv2.resize(source_mask.astype(np.float32), (imsize,imsize))>0, 1.0, 0.0)
                target_image = cv2.resize(target_image, (imsize,imsize))
                target_mask = np.where(cv2.resize(target_mask.astype(np.float32), (imsize,imsize))>0, 1.0, 0.0)

                # cp
                if cp_type == 'cp':
                    image, mask = cp(source_image, target_image, source_mask, target_mask)
                elif cp_type == 'cpSimple':
                    image, mask = cp_simple(source_image, target_image, source_mask, target_mask)
                elif cp_type == 'inpaintingCP':
                    image, mask = inpainting_cp(source_image, target_image, source_mask, target_mask)
                elif cp_type == 'tumorCP':
                    image, mask = tumor_cp(source_image, target_image, source_mask, target_mask)

            # save
            mask = mask * 255
            cv2.imwrite(fr'{path}/synthesized/{dataset}/{cp_type}/{count}.png', image)
            cv2.imwrite(fr'{path}/synthesized/{dataset}/{cp_type}/{count}_mask.png', mask)

            # metric
            image_torch = torch.einsum('bhwc -> bchw', torch.from_numpy(image[None,...]).to(dtype=torch.uint8))
            fid = metric.get_fid(None, image_torch, fid)
            inception = metric.get_is(image_torch, inception)
            
            # process bar
            count += 1
            pbar.update(1)
            time.sleep(0.05)
    
    # metric
    for image_name in image_list:
        if dataset == 'glas2015':
            image = cv2.imread(fr'{dataset_path}/{image_name}', cv2.IMREAD_COLOR)
        elif dataset == 'kumar':
            image = tifffile.imread(fr'{dataset_path}/train/Images/{image_name}')
        elif dataset == 'cpm17':
            image = cv2.imread(fr'{dataset_path}/train/Images/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.resize(image, (imsize,imsize))
        image_torch = torch.einsum('bhwc -> bchw', torch.from_numpy(image[None,...]).to(dtype=torch.uint8))
        fid = metric.get_fid(image_torch, None, fid)

    # save configuration
    configuration = {'cp_type':cp_type, 'n_samples':n_samples, 'dataset':dataset, 'imsize':imsize, 'self_mix':self_mix, 'fid':fid.compute().item(), 'is':inception.compute()[0].item()}
    misc.save_yaml(fr'{path}/synthesized/{dataset}/{cp_type}/configuration.yaml', configuration)
    del fid, inception
    
    

if __name__ == '__main__':
    run(cp_type='cp', n_samples=5000, imsize=512, dataset='glas2015')
    run(cp_type='cpSimple', n_samples=5000, imsize=512, dataset='glas2015')
    run(cp_type='inpaintingCP', n_samples=5000, imsize=512, dataset='glas2015')
    run(cp_type='tumorCP', n_samples=5000, imsize=512, dataset='glas2015')

    run(cp_type='cp', n_samples=5000, imsize=512, dataset='kumar')
    run(cp_type='cpSimple', n_samples=5000, imsize=512, dataset='kumar')
    run(cp_type='inpaintingCP', n_samples=5000, imsize=512, dataset='kumar')
    run(cp_type='tumorCP', n_samples=5000, imsize=512, dataset='kumar')

    run(cp_type='cp', n_samples=5000, imsize=512, dataset='cpm17')
    run(cp_type='cpSimple', n_samples=5000, imsize=512, dataset='cpm17')
    run(cp_type='inpaintingCP', n_samples=5000, imsize=512, dataset='cpm17')
    run(cp_type='tumorCP', n_samples=5000, imsize=512, dataset='cpm17')