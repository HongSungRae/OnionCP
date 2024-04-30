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
import copy

# local
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(path)

from utils import misc, metric, monuseg
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
    assert dataset in ['glas2015', 'kumar', 'cpm17', 'monuseg']
    assert (imsize >= 256) and (imsize%(2**4) == 0)
    if cp_type == 'onionCP':
        self_mix = True
    misc.seed_everything(10)

    # make save path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, ".."))
    misc.make_dir(fr'{path}/synthesized/{dataset}/{cp_type}')

    # load data info
    dataset_path = misc.open_yaml(fr'{path}/configuration.yaml')[dataset]

    # synthesize new samples
    count = 2122
    inception = None
    fid = None
    with tqdm(total=n_samples) as pbar:
        while count < n_samples+1:
            if dataset in ['glas2015']:
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
                if dataset in ['kumar', 'cpm17']:
                    image_list = os.listdir(fr'{dataset_path}/train/Images')
                else:
                    image_list = os.listdir(fr'{dataset_path}/train/Tissue Images')

                if self_mix:
                    source_image_name = random.sample(image_list, 1)[0]
                    target_image_name = source_image_name
                else:
                    source_image_name, target_image_name = random.sample(image_list, 2)
                
                if dataset == 'kumar':
                    source_image = tifffile.imread(fr'{dataset_path}/train/Images/{source_image_name}')
                    target_image = tifffile.imread(fr'{dataset_path}/train/Images/{target_image_name}')
                    source_mask = io.loadmat(fr'{dataset_path}/train/Labels/{source_image_name.split(".")[0]}.mat')['inst_map']
                    target_mask = io.loadmat(fr'{dataset_path}/train/Labels/{target_image_name.split(".")[0]}.mat')['inst_map']
                elif dataset == 'cpm17':
                    source_image = cv2.imread(fr'{dataset_path}/train/Images/{source_image_name}', cv2.IMREAD_COLOR)
                    target_image = cv2.imread(fr'{dataset_path}/train/Images/{target_image_name}', cv2.IMREAD_COLOR)
                    source_mask = io.loadmat(fr'{dataset_path}/train/Labels/{source_image_name.split(".")[0]}.mat')['inst_map']
                    target_mask = io.loadmat(fr'{dataset_path}/train/Labels/{target_image_name.split(".")[0]}.mat')['inst_map']
                elif dataset == 'monuseg':
                    source_image = tifffile.imread(fr'{dataset_path}/train/Tissue Images/{source_image_name}')
                    target_image = tifffile.imread(fr'{dataset_path}/train/Tissue Images/{target_image_name}')
                    source_mask = monuseg.load_mask(fr'{dataset_path}/train/Annotations', source_image_name.split(".")[0], source_image.shape)
                    target_mask = monuseg.load_mask(fr'{dataset_path}/train/Annotations', target_image_name.split(".")[0], target_image.shape)
                

                # try CP
            try:
                if cp_type == 'onionCP':
                    # resize image only
                    source_image = cv2.resize(source_image, (imsize,imsize))
                    target_image = cv2.resize(target_image, (imsize,imsize))

                    # select paste objects
                    max_objs = {'kumar':60, 'cpm17':40, 'glas2015':6}[dataset]
                    num_objects = min(random.randint(1, int((len(np.unique(source_mask))-1)/2)) * 2, max_objs)
                    selected_objects = random.sample(np.unique(source_mask).tolist()[1:], num_objects)
                    mask = np.where(cv2.resize(source_mask.astype(np.float64), (imsize,imsize))>0, 1, 0)
                    source_mask = np.where(np.isin(source_mask, selected_objects), source_mask, 0)
                    image = copy.deepcopy(target_image)
                    changed_mask = np.zeros((imsize,imsize))
                    
                    # compute area of each objects. sort ascending order
                    area_dict = {}
                    for value in selected_objects:
                        area_dict[value] = np.sum(source_mask==value)
                    area_dict_items = sorted(area_dict.items(), key=lambda x: x[1], reverse=False)

                    # check area size similarity and do CP
                    completed = False
                    for i in tqdm(range(int(len(area_dict_items)/2)), desc=f' Generating {count}th sample...'):
                        object_1_key, object_1_area = area_dict_items[i]
                        object_2_key, object_2_area = area_dict_items[i+1]
                        if object_2_area * 1.2 >= object_1_area:
                            completed = True
                            mask_1 = np.where(source_mask==object_1_key, 1, 0)
                            mask_1 = np.where(cv2.resize(mask_1.astype(np.float32),(imsize,imsize))>0, 1, 0)
                            mask_2 = np.where(source_mask==object_2_key, 1, 0)
                            mask_2 = np.where(cv2.resize(mask_2.astype(np.float32),(imsize,imsize))>0, 1, 0)
                            synthesized_image_1, slim_mask_1, original_mask_1 = onion_cp(source_image, target_image, mask_1, mask_2, dataset)
                            synthesized_image_2, slim_mask_2, original_mask_2 = onion_cp(source_image, target_image, mask_2, mask_1, dataset)
                            image = image*(1-slim_mask_1[...,None]) + synthesized_image_1*slim_mask_1[...,None]
                            image = image*(1-slim_mask_2[...,None]) + synthesized_image_2*slim_mask_2[...,None]
                            changed_mask += original_mask_1.astype(np.uint8)
                            changed_mask += original_mask_2.astype(np.uint8)
                    image = image.astype(np.uint8)
                    cv2.imwrite(fr'{path}/synthesized/{dataset}/{cp_type}/{count}_changed.png', changed_mask*255)

                    if not completed:
                        raise NotImplementedError('This sample is not suitable for augmentation.')
                else:
                    # select pasted objects
                    num_objects = random.randint(1, len(np.unique(source_mask))-1)
                    selected_objects = np.array(random.sample(np.unique(source_mask).tolist()[1:], num_objects))
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
                if dataset == 'kumar':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = mask * 255
                cv2.imwrite(fr'{path}/synthesized/{dataset}/{cp_type}/{count}.png', image)
                cv2.imwrite(fr'{path}/synthesized/{dataset}/{cp_type}/{count}_mask.png', mask)

                # metric
                # image_torch = torch.einsum('bhwc -> bchw', torch.from_numpy(image[None,...]).to(dtype=torch.uint8))
                # fid = metric.get_fid(None, image_torch, fid)
                # inception = metric.get_is(image_torch, inception)
            
                # process bar
                count += 1
                pbar.update(1)
                time.sleep(0.05)
            except Exception as e:
                print(e)
                # print(e.__traceback__.tb_lineno)
                # print(e, num_objects, np.unique(source_mask).tolist()[1:], source_image_name)

    
    # metric
    # for image_name in image_list:
    #     if dataset == 'glas2015':
    #         image = cv2.imread(fr'{dataset_path}/{image_name}', cv2.IMREAD_COLOR)
    #     elif dataset == 'kumar':
    #         image = tifffile.imread(fr'{dataset_path}/train/Images/{image_name}')
    #     elif dataset == 'cpm17':
    #         image = cv2.imread(fr'{dataset_path}/train/Images/{image_name}', cv2.IMREAD_COLOR)
    #     image = cv2.resize(image, (imsize,imsize))
    #     image_torch = torch.einsum('bhwc -> bchw', torch.from_numpy(image[None,...]).to(dtype=torch.uint8))
    #     fid = metric.get_fid(image_torch, None, fid)

    # save configuration
    configuration = {'cp_type':cp_type, 'n_samples':n_samples, 'dataset':dataset, 'imsize':imsize, 'self_mix':self_mix, 'fid':fid.compute().item(), 'is':inception.compute()[0].item()}
    misc.save_yaml(fr'{path}/synthesized/{dataset}/{cp_type}/configuration.yaml', configuration)
    del fid, inception
    
    

if __name__ == '__main__':
    # run(cp_type='cp', n_samples=5000, imsize=512, dataset='kumar')
    # run(cp_type='cpSimple', n_samples=5000, imsize=512, dataset='kumar')
    # run(cp_type='inpaintingCP', n_samples=5000, imsize=512, dataset='kumar')
    # run(cp_type='tumorCP', n_samples=5000, imsize=512, dataset='kumar')
    # print('kumar')
    # run(cp_type='onionCP', n_samples=5000, imsize=512, dataset='kumar')

    # run(cp_type='cp', n_samples=5000, imsize=512, dataset='cpm17')
    # run(cp_type='cpSimple', n_samples=5000, imsize=512, dataset='cpm17')
    # run(cp_type='inpaintingCP', n_samples=5000, imsize=512, dataset='cpm17')
    # run(cp_type='tumorCP', n_samples=5000, imsize=512, dataset='cpm17')
    run(cp_type='onionCP', n_samples=5000, imsize=512, dataset='cpm17')

    # run(cp_type='cp', n_samples=5000, imsize=512, dataset='monuseg')
    # run(cp_type='cpSimple', n_samples=5000, imsize=512, dataset='monuseg')
    # run(cp_type='inpaintingCP', n_samples=5000, imsize=512, dataset='monuseg')
    # run(cp_type='tumorCP', n_samples=5000, imsize=512, dataset='monuseg')
    # run(cp_type='onionCP', n_samples=50, imsize=256, dataset='monuseg')

    # run(cp_type='cp', n_samples=5000, imsize=512, dataset='glas2015')
    # run(cp_type='cpSimple', n_samples=5000, imsize=512, dataset='glas2015')
    # run(cp_type='inpaintingCP', n_samples=5000, imsize=512, dataset='glas2015')
    # run(cp_type='tumorCP', n_samples=5000, imsize=512, dataset='glas2015')
    # run(cp_type='onionCP', n_samples=50, imsize=256, dataset='glas2015')