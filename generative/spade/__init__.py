# package
from generative.spade import discriminator, encoder, ganloss, generator, spade_resblk, spade
__all__ = ['discriminator', 'encoder', 'ganloss', 'generator', 'spade_resblk', 'spade']

# library
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import numpy as np


# local
from .generator import SPADEGenerator
from .discriminator import SPADEDiscriminator, PatchDiscriminator
from .ganloss import GANLoss, GeneratorLoss, DiscrimonatorLoss
from utils import misc
from datasets.glas2015 import GlaS2015
from .resnet import GenertorResNet





def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    # elif classname.find('BatchNorm2d') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)




class SPADETrainer:
    def __init__(self, path, args):
        assert torch.cuda.is_available()==True, 'No GPU available.'

        self.path = path
        self.args = args

        # save configuration
        configuration = vars(args)
        misc.save_yaml(fr'{path}/data/bank/gan/{args.exp_name_gan}/configuration.yaml', configuration)

        # init G and D
        ## Generator
        self.generator = SPADEGenerator(gan_input_size=args.gan_input_size,
                                        gan_hidden_size=args.gan_hidden_size,
                                        spade_resblk_kernel=args.spade_resblk_kernel,
                                        spade_filter=args.spade_filter,
                                        spade_kernel=args.spade_kernel,
                                        imsize=int(args.imsize_gan))
        # self.generator = GenertorResNet((3,256,256), 9)
        self.generator.apply(weights_init)
        
        ## Discriminator
        # self.discriminator = SPADEDiscriminator(args)
        if int(args.imsize_gan) in [256,128]:
            if args.dsize == 'small':
                self.discriminator = PatchDiscriminator(in_channels=[3+1,16,32,16],
                                                        out_channels=[16,32,16,3],
                                                        kernel_size=[4,4,4,4],
                                                        stride=[2,2,2,2],
                                                        padding=[2,2,2,2])
            elif args.dsize == 'medium':
                self.discriminator = PatchDiscriminator(in_channels=[3+1,64,128,256,128],
                                                        out_channels=[64,128,256,128,3],
                                                        kernel_size=[3,3,3,3,3],
                                                        stride=[2,2,2,2,2],
                                                        padding=[4,8,8,7,4])
            else:
                self.discriminator = PatchDiscriminator(in_channels=[3+1,64,128,256,128,64],
                                                        out_channels=[64,128,256,128,64,3],
                                                        kernel_size=[3,3,3,3,3,3],
                                                        stride=[2,2,2,2,2,2],
                                                        padding=[4,8,8,7,4,4])
        elif int(args.imsize_gan) == 512:
            self.discriminator = PatchDiscriminator(in_channels=[4,16,64,32,16],
                                                    out_channels=[16,64,32,16,3],
                                                    kernel_size=[4,4,4,4,4],
                                                    stride=[2,2,2,2,2],
                                                    padding=[2,2,2,2,2])
        self.discriminator.apply(weights_init)

        ## cuda
        if args.distributed:
            self.generator = torch.nn.DataParallel(self.generator).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
        else:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        ## Loss
        # self.criterion = GANLoss(gan_mode='original')
        self.criterion_gen = GeneratorLoss(fm_loss=args.fm_loss, return_all=True)
        self.criterion_dis = DiscrimonatorLoss(lam=args.lam_dis, return_all=True)

        # dataloader
        self.train_dataloader, self.test_dataloader = self.get_dataloader()
        print(f'\n Train loader : {args.batch_size_gan} batch X {len(self.train_dataloader)} iters')
        print(f'\n Test loader : 4 batch X {len(self.test_dataloader)} iters')

    def get_dataloader(self):
        if self.args.dataset_gan == 'GlaS2015':
            train_dataset = GlaS2015(path=self.path,
                                     original=True,
                                     exp_name_cp=self.args.exp_name_cp,
                                     imsize=self.args.imsize_gan)
            test_dataset = GlaS2015(path=self.path,
                                    split='test',
                                    imsize=self.args.imsize_gan)
        elif self.args.dataset_gan == 'CRAG':
            pass
        elif self.args.dataset_gan == 'Seegene':
            pass

        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size_gan, drop_last=False, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, drop_last=True, shuffle=True)
        return train_dataloader, test_dataloader

    def train(self):
        # init
        ## optimizer
        optim_gen = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_gen, betas=(0.0, 0.999))
        optim_dis = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr_dis, betas=(0.0, 0.999))

        ## z
        # mean_std = misc.open_yaml(fr'{self.path}/data/bank/encoder/{self.args.exp_name_encoder}/mean_std.yaml')
        # mean = mean_std['']

        ## log
        loss_G_list = []
        loss_D_list = []
        loss_fm_list = []
        loss_fake_G_list = []
        loss_fake_D_list = []
        loss_real_D_list = []
        

        ## etc
        # scaler = GradScaler()
        # scaler_gen = GradScaler()
        # scaler_dis = GradScaler()
        # torch.backends.cudnn.benchmark = True

        # train
        for epoch in tqdm(range(self.args.epochs_gan), desc=' Alert : Training SPADE...'):
            self.generator.train()
            self.discriminator.train()
            for i, (img, mask, grade) in enumerate(self.train_dataloader):
                img, mask = img.cuda(), mask.cuda()
                with autocast():
                    noise = torch.randn(img.shape[0], 256).cuda()
                    fake_img = self.generator(noise, mask)

                    pred_fake = self.discriminator(fake_img, mask)
                    # layer_outputs_fake = self.discriminator.layer_outputs
                    # self.discriminator.layer_outputs = {}

                    pred_real = self.discriminator(img, mask)
                    # layer_outputs_real = self.discriminator.layer_outputs

                    # loss_G, loss_fake_G, loss_fm = self.criterion_gen(pred_fake, layer_outputs_real, layer_outputs_fake)
                    loss_G = self.criterion_gen(pred_fake) * self.args.lam_gen
                    loss_D, loss_real_D, loss_fake_D  = self.criterion_dis(pred_real, pred_fake)

                    # loss_G = self.criterion(pred_fake, target_is_real=True, for_discriminator=False)
                    # loss_D = self.criterion(pred_real, target_is_real=True)*0.5 + self.criterion(pred_fake, target_is_real=False)

                # loss_G = self.criterion(pred_fake, True)
                # loss_D = loss_D_fake + loss_D_real*0.5
                
                # optim_gen.zero_grad()
                # scaler_gen.scale(loss_G).backward()
                # scaler_gen.step(optim_gen)
                # scaler_gen.update() 

                # optim_dis.zero_grad()
                # scaler_dis.scale(loss_D).backward()
                # scaler_dis.step(optim_dis)
                # scaler_dis.update()
                
                optim_dis.zero_grad()
                loss_D.backward(retain_graph=True)
                optim_dis.step()

                optim_gen.zero_grad()
                loss_G.backward()
                optim_gen.step()

                # scaler.scale(loss_G).backward(retain_graph=True)
                # scaler.scale(loss_D).backward()
                # scaler.step(optim_gen)
                # scaler.step(optim_dis)
                # scaler.update()

                loss_G_list.append(loss_G.detach().cpu().item())
                loss_D_list.append(loss_D.detach().cpu().item())
                # loss_fm_list.append(10*loss_fm.detach().cpu().item())
                loss_fake_D_list.append(loss_fake_D.detach().cpu().item())
                loss_real_D_list.append(loss_real_D.detach().cpu().item())
                # loss_fake_G_list.append(loss_fake_G.detach().cpu().item())
            
            if (epoch+1)%20 == 0:
                ## show sythesized images
                del img, mask, noise, grade, pred_fake, pred_real, loss_D, loss_G#, loss_fake_D, loss_fake_G, loss_real_D, loss_fm
                optim_dis.zero_grad()
                optim_gen.zero_grad()
                torch.cuda.empty_cache()
                self.generator.eval()
                with torch.no_grad():
                    plt.cla()
                    plt.figure(figsize=(20,20))
                    count = 0
                    for _, (img, mask, grade) in enumerate(self.test_dataloader):
                        noise = torch.randn(img.shape[0], 256).cuda()
                        img, mask = img.cuda(), mask.cuda()
                        fake_imgs = self.generator(noise, mask)
                        fake_imgs = fake_imgs.detach().cpu()
                        fake_imgs = (fake_imgs*0.5 + 0.5)*255.0
                        for j in range(fake_imgs.shape[0]):
                            count += 1
                            plt.subplot(2,2,j+1)
                            plt.imshow(np.einsum('c...->...c', fake_imgs[j].numpy().astype(np.uint8)))
                            if count == 4:
                                break
                        if count == 4:
                            break
                    plt.savefig(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/{epoch+1}.png', format='png')
                del img, mask, fake_imgs
                torch.cuda.empty_cache()

                ## save G
                misc.save_model(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/generator_{epoch+1}.pt', 
                                self.generator,
                                self.args.distributed)
                
            ## plot loss D
            plt.cla()
            plt.figure(figsize=(20,10))
            plt.plot(loss_real_D_list, label='Loss on REAL')
            plt.plot(loss_fake_D_list, label='Loss on FAKE')
            plt.xlabel('Iter')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/loss_D.png', format='png')

            ## plot loss D&G
            plt.cla()
            plt.figure(figsize=(20,10))
            plt.plot(loss_D_list, label='Discriminator Loss')
            plt.plot(loss_G_list, label='Generator Loss')
            plt.xlabel('Iter')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/loss.png', format='png')

            ## plot loss G
            if self.args.fm_loss:
                plt.cla()
                plt.figure(figsize=(20,10))
                plt.plot(loss_fm_list, label='FM Loss')
                plt.plot(loss_fake_G_list, label='Loss on FAKE')
                plt.xlabel('Iter')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/loss_G_{epoch+1}.png', format='png')
        
        # when training ends
        ## D save
        misc.save_model(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/discriminator_{epoch+1}.pt',
                        self.discriminator,
                        self.args.distributed)

        ## save log
        misc.save_yaml(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/loss.yaml', {'generator':loss_G_list, 'discriminator':loss_D_list})
        misc.save_yaml(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/loss_D.yaml', {'loss_real_D':loss_real_D_list, 'loss_fake_D':loss_fake_D_list})
        if self.args.fm_loss:
            misc.save_yaml(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/loss_G.yaml', {'loss_fake_G':loss_fake_G_list, 'loss_fm':loss_fm_list})


    def inference(self):
        misc.make_dir(fr'{self.path}/data/bank/gan/{self.args.exp_name_gan}/samples')
        with torch.no_grad():
            pass


    def __call__(self, mode='train'):
        if mode == 'train':
            self.train()
            self.inference()
        else:
            self.inference()