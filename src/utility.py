import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
# import scipy.misc as misc
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''


        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        
        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        # self.n_processes = 8


    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        if self.args.save_results:
            os.makedirs('{}/results'.format(self.dir), exist_ok=True)
            filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
            
            postfix = ('SR', 'BICUBIC', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].data.mul(255 / self.args.rgb_range)
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                # tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                #misc.imsave('{}{}.png'.format(filename, p), ndarr)
                imageio.imsave('{}{}.png'.format(filename, p), ndarr)
                # self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

    # def save_results_nopostfix(self, filename, save_list, scale):
    #     #print(filename)
    #     if self.args.degradation == 'BI':
    #         filename = filename.replace("LRBI", self.args.save)
    #     elif self.args.degradation == 'BD':
    #         filename = filename.replace("LRBD", self.args.save)
        
    #     filename = '{}/{}/x{}/{}'.format(self.dir, self.args.testset, scale, filename)
    #     postfix = ('SR', 'LR', 'HR')
    #     for v, p in zip(save_list, postfix):
    #         normalized = v[0].data.mul(255 / self.args.rgb_range)
    #         ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
    #         # tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    #         #misc.imsave('{}.png'.format(filename), ndarr)
    #         imageio.imsave('{}.png'.format(filename), ndarr)
    #         # self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    # if hr.nelement() == 1: return 0
    diff = (sr - hr) / rgb_range
    
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, my_model):
    '''
        make optimizer
    '''
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer, epoch=-1):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma,
            last_epoch=epoch
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('-')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma,
            last_epoch=epoch
        )

    return scheduler

