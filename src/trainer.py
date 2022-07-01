import os
import math
from decimal import Decimal
from unittest import load_tests

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            self.scheduler = utility.make_scheduler(args, self.optimizer, epoch=len(ckp.log)-1)
            # 直接 step 会出问题
            # for _ in range(len(ckp.log)): 
            #     self.scheduler.step()

        else:
            self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        self.ckp.write_log(
            'Epoch[{}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        epoch_loss = 0.0
        iter_num = 0
        with tqdm(total=len(self.loader_train), desc='Epoch[{}]'.format(epoch)) as pbar:
            for batch, (lr, hr, _) in enumerate(self.loader_train):
                lr, hr = self.prepare(lr, hr)
                timer_data.hold()
                timer_model.tic()

                self.optimizer.zero_grad()
                if not self.args.random_output:
                    # 不随机跳出，每一次都跑完所有的 resgroups
                    sr = self.model(lr, 0, idx=self.args.n_resgroups)
                else:
                    sr = self.model(lr, 0)
                loss = self.loss(sr, hr)
                if loss.item() < self.args.skip_threshold * self.error_last:
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    iter_num += 1
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        batch + 1, loss.item()
                    ))

                timer_model.hold()

                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release()))

                timer_data.tic()
                pbar.update(1)

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()
        print("===> Epoch[{}-train](complete): Loss: {:.4f}\n".format(epoch, epoch_loss / iter_num))
        # if not self.args.test_only:
        #     self.ckp.save(self, epoch, is_best=False)

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.scheduler.last_epoch
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        for idx_scale, scale in enumerate(self.scale):
            sr_psnr_list = [0. for i in range(self.args.n_resgroups + 1)]
            eval_acc = 0.
            bicubic_acc =0.
            self.loader_test.dataset.set_scale(idx_scale)
            # print("len(self.loader_test):{}".format(len(self.loader_test)))
            with tqdm(total=len(self.loader_test), desc='Epoch[{}]'.format(epoch)) as pbar:
                for idx_img, (lr, hr, lr_upscale, filename) in enumerate(self.loader_test):
                    filename = filename[0]
                    lr, hr, lr_upscale = self.prepare(lr, hr, lr_upscale)
                    # 只测了从 0~n_resgroups-1 跳出
                    for idx in range(self.args.n_resgroups):
                        sr = self.model(lr, idx_scale, idx)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        sr_psnr_list[idx] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                    # 从 n_resgroups 跳出
                    sr = self.model(lr, idx_scale, self.args.n_resgroups)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]

                    eval_acc += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    
                    bicubic_acc += utility.calc_psnr(
                        lr_upscale, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    save_list.extend([lr_upscale, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)
                        # self.ckp.save_results_nopostfix(filename, save_list, scale)
                    pbar.update(1)
            sr_psnr_list[-1] = eval_acc  # 最后一层的
            if self.args.valid:
                self.ckp.write_log("===> Evaluation[{} x{}]:\nBICUBIC PSNR: {:4f} dB\nSR PSNR:".format(self.args.data_test, scale, bicubic_acc / len(self.loader_test)))
            else:
                self.ckp.write_log("===> Test[{} x{}]:\nBICUBIC PSNR: {:4f} dB\nSR PSNR:".format(self.args.data_test, scale, bicubic_acc / len(self.loader_test)))
            for idx in range(self.args.n_resgroups + 1):
                self.ckp.write_log("{:2} res_groups: {:.4f} dB".format(idx, sr_psnr_list[idx] / len(self.loader_test)))
            self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
            best = self.ckp.log.max(0)
            if self.args.valid:
                self.ckp.write_log(
                    '===> Evaluation[{} x{}]: BICUBIC PSNR: {:.4f} dB, SR PSNR: {:.4f} dB (Best: {:.4f} dB @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        bicubic_acc / len(self.loader_test),
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
            else:
                self.ckp.write_log(
                    '===> Test[{} x{}]: BICUBIC PSNR: {:.4f} dB, SR PSNR: {:.4f} dB (Best: {:.4f} dB @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        bicubic_acc / len(self.loader_test),
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log('Total time: {:.2f}s'.format(timer_test.toc(), refresh=True))


        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
            self.ckp.write_log("Saving checkpoint!\n")
        
        torch.set_grad_enabled(True)
        

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': 
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch > self.args.epochs

