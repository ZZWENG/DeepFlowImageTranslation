from torch.optim import Adam
from itertools import chain
from collections import OrderedDict
import os
import real_nvp_module.array_util as util
from real_nvp_module.real_nvp import RealNVP
from gan_utils import *

init_gain = 0.02
use_sigmoid = True
ndf = 64
lambda_A = 10.0
lambda_B = 10.0
lambda_idt = 0.5
pool_size = 50
num_scales = 2


class FlowGANTrainer(object):
    def __init__(self, nc, learning_rate, device='cpu', gpu_ids=[], is_train=True, num_blocks=4):
        super(FlowGANTrainer, self).__init__()
        self.device = device
        self.gpu_ids = gpu_ids
        self.is_train = is_train

        # Discriminator, Generator
        self.netD_A = define_D(nc, ndf, use_sigmoid, init_gain, gpu_ids=self.gpu_ids).to(device)
        self.netD_B = define_D(nc, ndf, use_sigmoid, init_gain, gpu_ids=self.gpu_ids).to(device)
        self.netG = RealNVP(in_channels=nc, num_scales=num_scales, num_blocks=num_blocks).to(device)

        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netG.to(gpu_ids[0])
            print('netG(RealNVP) to: GPU', gpu_ids[0])

        # Optimizers
        self.optimizer_dis = Adam(
            chain(self.netD_A.parameters(), self.netD_B.parameters()), learning_rate,
            betas=(0.5, 0.999), weight_decay=0.0005)
        self.optimizer_gen = Adam(self.netG.parameters(), 0.0001, eps=1e-6)

        # Loss functions
        self.criterionGAN = GANLoss(use_lsgan=False).to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        self.model_names = ['G', 'D_A', 'D_B']
        self.loss_names = ['D_A', 'G_A', 'idt_A', 'D_B', 'G_B', 'idt_B']
        self.nc = nc
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)
        self.space_depth_factor = 2

    def set_input(self, a, b):
        self.real_A = a
        self.real_B = b

    def forward(self):
        self.fake_B, kld1 = self.netG.forward(self.real_A)
        self.rec_A = self.netG.backward(self.fake_B)

        real_B = util.space_to_depth(self.real_B, self.space_depth_factor)
        self.fake_A = self.netG.backward(real_B)
        self.rec_B, kld2 = self.netG.forward(self.fake_A)

        self.fake_B = util.depth_to_space(self.fake_B, self.space_depth_factor)
        self.rec_B = util.depth_to_space(self.rec_B, self.space_depth_factor)
        return kld1, kld2

    def optimize_parameters(self):
        # forward
        kld1, kld2 = self.forward()

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_gen.zero_grad()
        self.backward_G()
        self.optimizer_gen.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_dis.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_dis.step()
        return kld1, kld2

    def backward_G(self):
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A, _ = self.netG.forward(self.real_B)
            self.idt_A = util.depth_to_space(self.idt_A, self.space_depth_factor)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            # G_B should be identity if real_A is fed.
            real_A = util.space_to_depth(self.real_A, self.space_depth_factor)
            self.idt_B = self.netG.backward(real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, epoch, save_dir):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if isinstance(net, torch.nn.DataParallel):
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, epoch, save_dir):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
