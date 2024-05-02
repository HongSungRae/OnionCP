import torch
from torch.nn import Module, Conv2d
from torch.nn.functional import relu
from torch.nn.utils import spectral_norm

from .spade import SPADE

class SPADEResBlk(Module):
    def __init__(self, spade_resblk_kernel, spade_filter, spade_kernel, k_in, k_out, skip=False, spectral=True):
        super().__init__()
        kernel_size = spade_resblk_kernel
        self.skip = skip

        
        if self.skip:
            self.spade1 = SPADE(spade_filter, spade_kernel, k_in)
            self.conv1 = Conv2d(k_in, k_in, kernel_size=(kernel_size, kernel_size), padding=1)
            self.spade_skip = SPADE(spade_filter, spade_kernel, k_in)
            self.conv_skip = Conv2d(k_in, k_out, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
        else:
            self.spade1 = SPADE(spade_filter, spade_kernel, k_in)
            self.conv1 = Conv2d(k_in, k_in, kernel_size=(kernel_size, kernel_size), padding=1)

        self.spade2 = SPADE(spade_filter, spade_kernel, k_in)
        self.conv2 = Conv2d(k_in, k_out, kernel_size=(kernel_size, kernel_size), padding=1)

        if spectral:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            if skip:
                self.conv_skip = spectral_norm(self.conv_skip)
    
    def forward(self, x, seg):
        x_skip = x
    
        x = relu(self.spade1(x, seg))
        x = self.conv1(x)
        x = relu(self.spade2(x, seg))
        x = self.conv2(x)

        if self.skip:
            x_skip = relu(self.spade_skip(x_skip, seg))
            x_skip = self.conv_skip(x_skip)
        
        return x_skip + x