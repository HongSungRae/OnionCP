import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from sklearn import metrics
import cv2
import numpy as np
import tifffile
import os
import re
from tqdm import tqdm

# local
from utils import misc
'''
and : (1 1) -> 1, 0 otherwise
xor : (0 1) (1 0) -> 1, 0 otherwise
or : (0 0) -> 0, 1 otherwise
'''




def get_fid(img_real=None, img_fake=None, fid=None):
    '''
    img_real/fake : torch.uint8 type, (batch, 3, h, w)
    fid : FrechetInceptionDistance instance. Input when it is available
    '''
    assert (img_real is not None) or (img_fake is not None), 'You have to measure at least one component : real or fake'

    if fid is None:
        fid = FrechetInceptionDistance(feature=64)
    if img_real is not None:
        fid.update(img_real, real=True)
    if img_fake is not None:
        fid.update(img_fake, real=False)
    return fid



def get_is(img_fake, inception=None):
    '''
    img_fake : torch.uint8 type, (batch, 3, h, w)
    inception : InceptionScore instance. Input when it is available
    '''
    if inception is None:
        inception = InceptionScore()
    inception.update(img_fake)
    return inception



def func_get_fid_is(dataset, generate, imsize=512):
    '''
    = input =

    = output =
    retruns nothing
    but prints IS and FID
    '''
    assert dataset in []
    assert generate in ['cp', 'cpSimple', 'inpaintingCP', 'onionCP', 'tumorCP', 'ddpm', 'gan']
    # init
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_path = misc.open_yaml(fr'{path}/configuration.yaml')[dataset]
    inception = None
    fid = None

    # real image_list
    if dataset == 'glas2015':
        real_image_list = [item for item in os.listdir(fr'{dataset_path}') if "train" in item and "anno" not in item]
    elif dataset in ['kumar', 'cpm17']:
        real_image_list = os.listdir(fr'{dataset_path}/train/Images')
    elif dataset == 'monuseg':
        real_image_list = os.listdir(fr'{dataset_path}/train/Tissue Images')
    
    # FID, IS : Fake images
    pattern = r'\d+\.png'
    fake_image_list = [item for item in os.listdir(fr'{path}/synthesized/{dataset}/{generate}') if re.match(pattern, item)]
    for image_name in tqdm(fake_image_list, desc=' Fake image...'):
        image = cv2.imdecode(np.fromfile(fr'{path}/synthesized/{dataset}/{generate}/{image_name}', np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (imsize, imsize))
        image_torch = torch.einsum('bhwc -> bchw', torch.from_numpy(image[None,...]).to(dtype=torch.uint8))
        fid = get_fid(None, image_torch, fid)
        inception = get_is(image_torch, inception)

    # FID metric : Real images
    for image_name in real_image_list:
        if dataset == 'glas2015':
            image = cv2.imdecode(np.fromfile(fr'{dataset_path}/{image_name}', np.uint8), cv2.IMREAD_COLOR)
        elif dataset == 'kumar':
            image = tifffile.imread(fr'{dataset_path}/train/Images/{image_name}')
        elif dataset == 'cpm17':
            image = cv2.imdecode(np.fromfile(fr'{dataset_path}/train/Images/{image_name}', np.uint8), cv2.IMREAD_COLOR)
        elif dataset == 'monuseg':
            image = tifffile.imread(fr'{dataset_path}/train/Tissue Images/{image_name}')
        image = cv2.resize(image, (imsize,imsize))
        image_torch = torch.einsum('bhwc -> bchw', torch.from_numpy(image[None,...]).to(dtype=torch.uint8))
        fid = get_fid(image_torch, None, fid)

    # print
    print(f'{dataset}({generate}) : FID({fid.compute().item()}), IS({inception.compute().item()})')



def pre_processing(pred, target):
    '''
    It deletes background channel. And returns one-hot processed pred tensor.
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : pred : (b,w,h,c-1), one-hot to channel
             target : (b,w,h,c-1), one-hot to channel
             eps : 1e-5
    '''
    num_classes = pred.shape[-1]
    target = target[...,1:] # select all except background class # (b,w,h,c-1)
    pred = torch.argmax(pred,-1) # (b,w,h)
    pred = torch.nn.functional.one_hot(pred, num_classes=num_classes)[...,1:] # (b,w,h,c-1) one-hot, except background class
    return pred, target, 1e-5



def get_accuracy(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    = Caution =
    Accuracy might be bigger than you expected because of large TrueNegative.
    Accuracy is not a good metric for segmentation.
    '''
    pred, target, eps = pre_processing(pred, target)
    tp_and_tn = ~torch.logical_xor(pred, target) # It returns (0,0) & (1,1) cases
    fp_and_fn = ~tp_and_tn # It returns (0,1) & (1,0) cases
    return ((torch.sum(tp_and_tn))/(torch.sum(tp_and_tn + fp_and_fn) + eps)).item()


def get_recall(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    return (torch.sum(torch.logical_and(pred, target))/(torch.sum(target) + eps)).item()


def get_precision(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    return (torch.sum(torch.logical_and(pred, target))/(torch.sum(pred) + eps)).item()


def get_f1(pred, target):
    '''
    F1 Score is equivalent to IoU.
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    eps = 1e-5
    precision = get_precision(pred, target)
    recall = get_recall(pred, target)
    return 2*(precision*recall)/(precision+recall+eps)


def get_iou(pred, target):
    '''
    IoU is equivalent to F1 Score.
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    intersection = torch.logical_and(pred, target)
    union = torch.logical_or(pred, target)
    return (torch.sum(intersection)/(torch.sum(union)+ eps)).item()


def get_dice_coeff(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    intersection = torch.logical_and(pred, target) # True Positive
    return (2*(torch.sum(intersection))/(torch.sum(pred) + torch.sum(target) + eps)).item()


def get_mae(pred, target, except_background=True):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    if except_background:
        pred, target, eps = pre_processing(pred, target)
    else:
        pred = torch.argmax(pred, dim=-1)
        pred = torch.nn.functional.one_hot(pred, num_classes=target.shape[-1])
        eps = 1e-5
    bwhc = pred.shape[0] * pred.shape[1] * pred.shape[2] * pred.shape[3] # for batches, width, height and channel
    return (torch.sum(torch.abs(pred-target))).item()/(bwhc + eps)



def get_auroc(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, _ = pre_processing(pred, target) # pred : (b,w,h,c-1) one-hot, except background class
    auroc = 0
    pred, target = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
    for batch in range(pred.shape[0]): # for bactches
        for cls in range(pred.shape[-1]): # for classes
            y_true = target[batch,...,cls] # (w,h)
            y_pred = pred[batch,...,cls] # (w,h)
            auroc += metrics.roc_auc_score(y_true, y_pred,labels=2)
    return auroc/(pred.shape[0]*pred.shape[-1])



def get_aji(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    intersection = torch.logical_and(pred, target) # True Positive
    return (torch.sum(intersection)/(torch.sum(pred) + torch.sum(target) - torch.sum(intersection) + eps)).item()



if __name__ == '__main__':
    imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
    print(imgs_dist1.shape)