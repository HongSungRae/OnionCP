import torch
from torch.nn import Module, Linear, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import tanh, interpolate 
from .spade_resblk import SPADEResBlk

class SPADEGenerator(Module):
    def __init__(self, gan_input_size=256, gan_hidden_size=16384, spade_resblk_kernel=3, spade_filter=128, spade_kernel=3, imsize=512):
        super().__init__()
        assert int(imsize) in [128,256,512] 
        self.imsize = imsize
        self.linear = Linear(gan_input_size,8192)#, gan_hidden_size)
        self.spade_resblk1 = SPADEResBlk(spade_resblk_kernel, spade_filter, spade_kernel, 512, 512, True) # 1024 1024
        self.spade_resblk2 = SPADEResBlk(spade_resblk_kernel, spade_filter, spade_kernel, 512, 512, True) # 1024 1024
        self.spade_resblk3 = SPADEResBlk(spade_resblk_kernel, spade_filter, spade_kernel, 512, 512, True) # 1024 1024
        self.spade_resblk4 = SPADEResBlk(spade_resblk_kernel, spade_filter, spade_kernel, 512, 256, True) # 1024 512
        self.spade_resblk5 = SPADEResBlk(spade_resblk_kernel, spade_filter, spade_kernel, 256, 128, True) # 512 256
        self.spade_resblk6 = SPADEResBlk(spade_resblk_kernel, spade_filter, spade_kernel, 128, 128, True) # 256 128
        self.spade_resblk7 = SPADEResBlk(spade_resblk_kernel, spade_filter, spade_kernel, 128, 64, True) # 128 64
        self.conv = spectral_norm(Conv2d(64, 3, kernel_size=(3,3), padding=1))

    def forward(self, x, seg):
        b, _, _, _ = seg.size()
        if self.imsize == 512:
            h, w = 4, 4
        elif self.imsize == 256:
            h, w = 2, 2
        else:
            h, w = 1, 1

        x = self.linear(x)
        x = x.view(b, -1, 4, 4) # (b,1024,4,4)

        x = interpolate(self.spade_resblk1(x, seg), size=(2*h, 2*w), mode='nearest') #; print(f'\n block1 : {torch.mean(x).item()}') # (b,1024,8,8)
        x = interpolate(self.spade_resblk2(x, seg), size=(4*h, 4*w), mode='nearest') #; print(f'\n block2 : {torch.mean(x).item()}')# (b,1024,16,16)
        x = interpolate(self.spade_resblk3(x, seg), size=(8*h, 8*w), mode='nearest') #; print(f'\n block3 : {torch.mean(x).item()}')# (b,1024,32,32)
        x = interpolate(self.spade_resblk4(x, seg), size=(16*h, 16*w), mode='nearest') #; print(f'\n block4 : {torch.mean(x).item()}')# (b,512,64,64)
        x = interpolate(self.spade_resblk5(x, seg), size=(32*h, 32*w), mode='nearest') #; print(f'\n block5 : {torch.mean(x).item()}')# (b,256,128,128)
        x = interpolate(self.spade_resblk6(x, seg), size=(64*h, 64*w), mode='nearest') #; print(f'\n block6 : {torch.mean(x).item()}')# (b,128,256,256)
        x = interpolate(self.spade_resblk7(x, seg), size=(128*h, 128*w), mode='nearest') #; print(f'\n block7 : {torch.mean(x).item()}')# (b,64,512,512)
        
        x = tanh(self.conv(x)) # (b,3,512,512)

        return x