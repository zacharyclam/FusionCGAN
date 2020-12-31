import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, generator, discriminator, criterions, metric_ftns, optimizerG, optimizerD, config, data_loader,
                 test_data_loader=None, eval_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(generator, discriminator, criterions, metric_ftns, optimizerG, optimizerD, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.test_data_loader = test_data_loader
        self.eval_data_loader = eval_data_loader

        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.adv_loss = criterions["mse_loss"]
        self.content_loss = criterions["l1_loss"]
        self.ssim_loss = criterions["ssim_loss"]

        self.train_metrics = MetricTracker("gen_loss", "dis_loss")
        self.test_metrics = MetricTracker("ssim_loss")
        # self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # add model structure to tensorboard
        # self.writer_gen.add_graph(self.generator, torch.ones(1, 8, 256, 256, device=self.device))
        # self.writer_dis.add_graph(self.discriminator, torch.ones(1, 3, 256, 256, device=self.device))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # batch_size = self.data_loader.batch_size
        # patch_size = self.data_loader.patch_size
        batch_size = 8
        patch_size = 32

        self.generator.train()
        self.discriminator.train()
        # self.train_metrics.reset()
        for batch_idx, (alpha_fg, alpha_bg, blur_fg, blur_bg, img_ref) in enumerate(self.data_loader):
            alpha_fg, alpha_bg, blur_fg, blur_bg, img_ref = alpha_fg.to(self.device), alpha_bg.to(
                self.device), blur_fg.to(self.device), blur_bg.to(self.device), img_ref.to(self.device)

            for p in self.discriminator.parameters():
                p.requires_grad = False

            input_concat = torch.cat([blur_fg, alpha_fg, blur_bg, alpha_bg], dim=1)

            gen_merge = self.generator(input_concat)
            gen_adv_loss = self.adv_loss(gen_merge, img_ref)
            gen_cont_loss = self.content_loss(gen_merge, img_ref)

            gen_loss = gen_cont_loss + 0.1 * gen_adv_loss

            self.optimizerG.zero_grad()
            gen_loss.backward()
            self.optimizerG.step()

            for p in self.discriminator.parameters():
                p.requires_grad = True

            real_label = torch.ones((batch_size, patch_size, patch_size), dtype=torch.float32).to(self.device)
            fake_label = torch.zeros((batch_size, patch_size, patch_size), dtype=torch.float32).to(self.device)

            fake_score = self.discriminator(gen_merge.detach())
            fake_score = self.adv_loss(fake_score, fake_label)

            real_score = self.discriminator(img_ref)
            real_score = self.adv_loss(real_score, real_label)

            dis_loss = (real_score + fake_score) * 0.5

            self.optimizerD.zero_grad()
            dis_loss.backward()
            self.optimizerD.step()
            # test
            self.writer_gen.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('gen_loss', gen_loss.item())
            self.train_metrics.update('dis_loss', dis_loss.item())

            self.writer_gen.add_scalar("gen_loss", gen_loss.item(), (epoch - 1) * self.len_epoch + batch_idx)
            self.writer_dis.add_scalar("dis_loss", dis_loss.item(), (epoch - 1) * self.len_epoch + batch_idx)
            #
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} gen_Loss: {:.6f}  dis_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    gen_loss.item(),
                    dis_loss.item()))
                self.writer_gen.add_image('output', make_grid(gen_merge.detach().cpu(), nrow=2, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_test:
            val_log = self._test_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _test_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        self.test_metrics.reset()
        SSIM = 0
        with torch.no_grad():
            for batch_idx, (alpha_fg, alpha_bg, blur_fg, blur_bg, img_ref) in enumerate(self.test_data_loader):
                alpha_fg, alpha_bg, blur_fg, blur_bg, img_ref = alpha_fg.to(self.device), alpha_bg.to(
                    self.device), blur_fg.to(self.device), blur_bg.to(self.device), img_ref.to(self.device)

                input_concat = torch.cat([blur_fg, alpha_fg, blur_bg, alpha_bg], dim=1)
                gen_merge = self.generator(input_concat)

                ssim = self.ssim_loss(gen_merge, img_ref)
                SSIM += ssim.item()

                self.writer_gen.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                self.writer_gen.add_scalar("ssim_loss", ssim.item(),
                                           (epoch - 1) * len(self.test_data_loader) + batch_idx)
                self.test_metrics.update('ssim_loss', ssim.item())
                # for met in self.metric_ftns:
                #     self.valid_metrics.update(met.__name__, met(output, target))
            # evaluation
            for batch_idx, (far_img, mask_far_img, near_img, mask_near_img) in enumerate(self.eval_data_loader):
                far_img, mask_far_img, near_img, mask_near_img = far_img.to(self.device), mask_far_img.to(
                    self.device), near_img.to(self.device), mask_near_img.to(self.device)

                input_concat = torch.cat([far_img, mask_far_img, near_img, mask_near_img], dim=1)
                gen_merge = self.generator(input_concat)

                self.writer_gen.add_image('output_test', make_grid(gen_merge.detach().cpu(), nrow=2, normalize=True))

        # print("epoch: ", epoch, " test_ssim:  ", SSIM / len(self.test_data_loader))

        # add histogram of model parameters to the tensorboard
        for name, p in self.generator.named_parameters():
            self.writer_gen.add_histogram(name, p, bins='auto')
        for name, p in self.discriminator.named_parameters():
            self.writer_dis.add_histogram(name, p, bins='auto')

        return self.test_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
