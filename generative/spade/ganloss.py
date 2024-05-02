import torch
import torch.nn as nn
import torch.nn.functional as F



class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real).to(self.device)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real).to(self.device)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                target_tensor = self.get_zero_tensor(input).to(self.device)
                if target_is_real:
                    minval = torch.min(input - 1, target_tensor)
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, target_tensor)
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        self.device = input.device
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)
        
    

class GeneratorLoss(nn.Module):
    def __init__(self, fm_loss=False, return_all=False, lam=0):
        super().__init__()
        self.return_all = return_all
        self.fm_loss = fm_loss
        self.lam = lam
        self.l1 = nn.L1Loss()

    def forward(self, pred_fake, layer_outputs_real=None, layer_outputs_fake=None):
        loss_fake = torch.mean((pred_fake-1)**2)
        if self.fm_loss:
            assert layer_outputs_real is not None, 'Please get layer_outputs from Discriminator.layer_outputs'
            assert layer_outputs_fake is not None, 'Please get layer_outputs from Discriminator.layer_outputs'
            loss_fm = 0
            for key in layer_outputs_fake.keys():
                loss_fm += self.l1(layer_outputs_real[key], layer_outputs_fake[key])
            if self.return_all:
                return loss_fake + self.lam*loss_fm, loss_fake, loss_fm
            else:
                return loss_fake + loss_fm
        else:
            return loss_fake



class DiscrimonatorLoss(nn.Module):
    def __init__(self, lam=1, return_all=False):
        super().__init__()
        self.return_all = return_all
        self.lam = lam

    def forward(self,pred_real, pred_fake):
        loss_real = torch.mean((pred_real-1)**2)
        loss_fake = torch.mean((pred_fake)**2)
        loss = self.lam*loss_real + loss_fake
        if self.return_all:
            return loss, loss_real, loss_fake
        else:
            return loss


if __name__ == '__main__':
    pass
    # discriminator = SPADEDiscriminator(None)
    # criterion_dis = DiscrimonatorLoss()
    # criterion_gen = GeneratorLoss(fm_loss=False)

    # img_real = nn.functional.tanh(torch.randn((4,3,512,512)))
    # img_fake = nn.functional.tanh(torch.randn((4,3,512,512)))
    # mask_real = torch.ones((4,1,512,512))
    # mask_fake = torch.ones((4,1,512,512))

    # pred_real = discriminator(img_real, mask_real)
    # layer_outputs_real = discriminator.layer_outputs
    # discriminator.layer_outputs = {}

    # pred_fake = discriminator(img_fake, mask_fake)
    # layer_outputs_fake = discriminator.layer_outputs

    # loss_dis = criterion_dis(pred_real, pred_fake)
    # loss_gen = criterion_gen(pred_fake, layer_outputs_real, layer_outputs_fake)
    
    # print(f'D Loss : {loss_dis.item()}')
    # print(f'G Loss : {loss_gen.item()}')





    # criterion_dis = DiscrimonatorLoss()

    # pred_real = nn.functional.sigmoid(torch.randn((4,1)))
    # pred_fake = nn.functional.sigmoid(torch.randn((4,1)))

    # loss_dis = criterion_dis(pred_real, pred_fake)
    # print(loss_dis)