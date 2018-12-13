import time
from optparse import OptionParser
import sys
import itertools
import numpy as np
import os
import torch
import glob
import pickle
from tqdm import tqdm

import torchvision.utils as vutils
import real_nvp_module.array_util as util

parser = OptionParser()
parser.add_option('--name', type=str, default='experiment')
parser.add_option('--batch_size', type=int, default=1)
parser.add_option('--datasetA', type=str, default='usps')
parser.add_option('--datasetB', type=str, default='mnist')
parser.add_option('--nc', type=int, default=3)
parser.add_option('--num_epochs', type=int, default=20)
parser.add_option('--learning_rate', type=float, default=0.005)
parser.add_option('--display', type=int, default=100)
parser.add_option('--save_img_freq', type=int, default=100)
parser.add_option('--gpu_ids', type=str, default='')
parser.add_option('--model', type=str, default='flowgan')
parser.add_option('--downsample', type=int, default=2)
parser.add_option('--save_one_file', type=int, default=1)
parser.add_option('--is_train', type=int, default=1)
parser.add_option('--start_from_latest_epoch', type=int, default=0)
parser.add_option('--save_epoch_per', type=int, default=5)
parser.add_option('--load_epoch', type=int, default=0)


def main(config):
    nc = config.nc
    name = config.name
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    gpu_ids = config.gpu_ids
    model = config.model
    downsample = config.downsample
    save_one_file = config.save_one_file
    is_train = config.is_train
    start_from_latest_epoch = config.start_from_latest_epoch
    save_epoch_per = config.save_epoch_per
    # GPU
    if config.gpu_ids:
        gpu_ids = eval(config.gpu_ids)
    device = 'cuda' if torch.cuda.is_available() and len(gpu_ids) > 0 else 'cpu'
    print('Training on device:', device)

    checkpoints_save_dir = os.path.join(os.path.curdir, 'logs', 'checkpoints',
                                        '{}_{}_batch{}'.format(model, name, batch_size))

    if not os.path.exists(checkpoints_save_dir):
        os.mkdir(checkpoints_save_dir)

    # Choose the model to train
    if model == 'flowgan':
        from modules.flowgan_trainer import FlowGANTrainer
        trainer = FlowGANTrainer
    elif model == 'cyclegan':
        from modules.cyclegan_trainer import CycleGANTrainer
        trainer = CycleGANTrainer
    else:
        raise Exception('Model {} does not exist!'.format(config.model))

    trainer = trainer(nc, learning_rate, device, gpu_ids)

    if not is_train:
        if config.load_epoch > 0:
            latest_epoch = config.load_epoch
        else:
            list_of_files = glob.glob('{}/*'.format(checkpoints_save_dir))
            latest_file = max(list_of_files, key=os.path.getctime)
            latest_epoch = int(latest_file.split('/')[-1].split('_')[0])
        trainer.load_networks(latest_epoch, checkpoints_save_dir)
        test(config, trainer, device, latest_epoch, model)
        print('Saved test results for epoch', latest_epoch)
        exit(0)
    if start_from_latest_epoch:
        list_of_files = glob.glob('{}/*'.format(checkpoints_save_dir))
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_epoch = int(latest_file.split('/')[-1].split('_')[0])
        trainer.load_networks(latest_epoch, checkpoints_save_dir)
    else:
        latest_epoch = -1

    d1 = config.datasetA
    d2 = config.datasetB
    if name.startswith('cityscapes') or name.startswith('apple2orange') or name.startswith('facades'):
        d1 = 'from_images_A'
        d2 = 'from_images_B'
    exec "import data.load_{} as dl_A".format(d1)
    exec "import data.load_{} as dl_B".format(d2)
    train_loader_a = dl_A.get_data_loader(config.batch_size, name, downsample)
    train_loader_b = dl_B.get_data_loader(config.batch_size, name, downsample)

    for epoch in range(config.num_epochs+1):
        if epoch <= latest_epoch:
            # print('Skipping', epoch, latest_epoch)
            continue
        epoch_start_time = time.time()
        print('Start training for epoch {}'.format(epoch))
        with tqdm(total=len(train_loader_a.dataset)) as progress_bar:
            for it, ((images_a, _), (images_b, _)) in enumerate(itertools.izip(train_loader_a, train_loader_b)):

                images_a = images_a.to(device)
                images_b = images_b.to(device)
                trainer.set_input(images_a, images_b)
                trainer.optimize_parameters()

                if it % config.display == 0:
                    losses = trainer.get_current_losses()
                    print('Current Loss: {}'.format(losses))

                if it % config.save_img_freq == 0:
                    if model == 'cyclegan':
                        fake_B = trainer.netG_A(images_a)
                        fake_A = trainer.netG_B(images_b)
                    else:
                        fake_B, _ = trainer.netG(images_a)
                        fake_A = trainer.netG.backward(util.space_to_depth(images_b, 2))
                        fake_B = util.depth_to_space(fake_B, 2)

                    if not save_one_file:
                        nrow = int(np.sqrt(batch_size))
                        vutils.save_image(fake_A,
                                          'sampling/fake_image_a_{}_{}_{}.png'.format(config.name, epoch, it),
                                          nrow=nrow)
                        vutils.save_image(fake_B,
                                          'sampling/fake_image_b_{}_{}_{}.png'.format(config.name, epoch, it),
                                          nrow=nrow)
                        vutils.save_image(images_a,
                                          'sampling/real_image_a_{}_{}_{}.png'.format(config.name, epoch, it),
                                          nrow=nrow)
                        vutils.save_image(images_b,
                                          'sampling/real_image_b_{}_{}_{}.png'.format(config.name, epoch, it),
                                          nrow=nrow)
                    else:
                        assert batch_size == 1
                        grid = torch.cat((images_a, fake_B, images_b, fake_A), 0)
                        vutils.save_image(grid,
                                          'sampling/result_{}_{}_{}_{}.png'.format(config.name, model, epoch, it),
                                          nrow=2)

                progress_bar.update(batch_size)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, config.num_epochs, time.time() - epoch_start_time))

            if (start_from_latest_epoch and model == 'flowgan') or (epoch % save_epoch_per == 0 and epoch > 0):
                # saving every 5 epochs
                try:
                    trainer.save_networks(epoch, checkpoints_save_dir)
                except:
                    print 'Cannot save network'


def test(config, trainer, device, epoch, model):
    name = config.name
    downsample = config.downsample

    if name.startswith('cityscapes') or name.startswith('apple2orange') or name.startswith('facades'):
        d1 = 'from_images_A'
        d2 = 'from_images_B'
    exec "import data.load_{} as dl_A".format(d1)
    exec "import data.load_{} as dl_B".format(d2)
    test_loader_a = dl_A.get_data_loader(1, name, downsample, is_train=False)
    test_loader_b = dl_B.get_data_loader(1, name, downsample, is_train=False)
    total_losses = {}
    with tqdm(total=len(test_loader_a.dataset)) as progress_bar:
        for it, ((images_a, _), (images_b, _)) in enumerate(itertools.izip(test_loader_a, test_loader_b)):

            images_a = images_a.to(device)
            images_b = images_b.to(device)

            trainer.set_input(images_a, images_b)
            trainer.forward()
            trainer.backward_G()
            trainer.backward_D_A()
            trainer.backward_D_B()
            fake_A = trainer.fake_A
            fake_B = trainer.fake_B
            rec_A = trainer.rec_A
            rec_B = trainer.rec_B

            losses = trainer.get_current_losses()
            total_losses[it] = losses

            if not os.path.exists('test_results'):
                os.mkdir('test_results')
            if it % 100 == 0:
                grid = torch.cat((images_a, fake_B, rec_A, images_b, fake_A, rec_B), 0)
                vutils.save_image(grid,
                                  'test_results/result_{}_{}_{}_{}.png'.format(config.name, model, epoch, it),
                                  nrow=3)

            path_fake_A = 'test_results/{}_{}_fakeA_{}'.format(config.name, model, epoch)
            path_fake_B = 'test_results/{}_{}_fakeB_{}'.format(config.name, model, epoch)
            path_real_A = 'test_results/{}_{}_realA_{}'.format(config.name, model, epoch)
            path_real_B = 'test_results/{}_{}_realB_{}'.format(config.name, model, epoch)
            if not os.path.exists(path_fake_A):
                os.mkdir(path_fake_A)
            if not os.path.exists(path_fake_B):
                os.mkdir(path_fake_B)
            if not os.path.exists(path_real_A):
                os.mkdir(path_real_A)
            if not os.path.exists(path_real_B):
                os.mkdir(path_real_B)
            vutils.save_image(images_a, '{}/{}.png'.format(path_real_A, it))
            vutils.save_image(images_b, '{}/{}.png'.format(path_real_B, it))
            vutils.save_image(fake_A, '{}/{}.png'.format(path_fake_A, it))
            vutils.save_image(fake_B, '{}/{}.png'.format(path_fake_B, it))
            progress_bar.update(1)
    mean_losses = {}
    for loss_name in ['D_A', 'G_A', 'idt_A', 'D_B', 'G_B', 'idt_B']:
        mean_losses[loss_name] = np.mean([d[loss_name] for d in total_losses.values()])
        print(loss_name, ':', mean_losses[loss_name])
    with open('test_results/{}_{}_loss_{}'.format(name, model, epoch), 'wb') as f:
        pickle.dump(mean_losses, f)


if __name__ == '__main__':
    (config, args) = parser.parse_args(sys.argv)
    main(config)
