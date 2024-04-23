import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore




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


if __name__ == '__main__':
    pass
    imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
    print(imgs_dist1.shape)