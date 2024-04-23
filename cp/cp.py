import numpy as np



def cp(img_source, img_target, mask_source, mask_target):
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
    
    img = np.zeros_like(img_source)
    img = img_source*mask_source[...,None] + img_target*(1-mask_source[...,None])
    mask = np.where((mask_source + mask_target) > 0, 1, 0)

    return img.astype(np.uint8), mask