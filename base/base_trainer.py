import os
from abc import abstractmethod
from pathlib import Path

import torch
from numpy import inf

from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, generator, discriminator, criterions, metric_ftns, optimizerG, optimizerD, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        if len(device_ids) > 1:
            self.generator = torch.nn.DataParallel(generator, device_ids=device_ids)
            self.discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

        self.criterions = criterions
        self.metric_ftns = metric_ftns
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            # self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            # self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # initialize checpoint
        os.mknod(self.checkpoint_dir / "checkpoint")

        # setup visualization writer instance                
        self.writer_gen = TensorboardWriter(os.path.join(config.log_dir, "generator"), self.logger,
                                            cfg_trainer['tensorboard'])

        self.writer_dis = TensorboardWriter(os.path.join(config.log_dir, "discrimiantor"), self.logger,
                                            cfg_trainer['tensorboard'])
        # self.writer_dreal = TensorboardWriter(os.path.join(config.log_dir, "d_real"), self.logger,
        #                                       cfg_trainer['tensorboard'])
        # self.writer_dfake = TensorboardWriter(os.path.join(config.log_dir, "d_fake"), self.logger,
        #                                       cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        netG = type(self.generator).__name__
        stateG = {
            'arch': netG,
            'epoch': epoch,
            'state_dict': self.generator.state_dict(),
            'optimizer': self.optimizerG.state_dict(),
            'config': self.config
        }
        filenameG = str(self.checkpoint_dir / 'checkpoint-netG-epoch{}.pth'.format(epoch))
        torch.save(stateG, filenameG)

        netD = type(self.discriminator).__name__
        stateD = {
            'arch': netD,
            'epoch': epoch,
            'state_dict': self.discriminator.state_dict(),
            'optimizer': self.optimizerD.state_dict(),
            'config': self.config
        }
        filenameD = str(self.checkpoint_dir / 'checkpoint-netD-epoch{}.pth'.format(epoch))
        torch.save(stateD, filenameD)

        self.logger.info("Saving checkpoint: {}   {}...".format(filenameG, filenameD))

        # write model info to checkpoint
        with open(self.checkpoint_dir / "checkpoint", "w") as f:
            line = "{}\n{}\n".format(filenameG, filenameD)
            f.write(line)

        if save_best:
            best_pathG = str(self.checkpoint_dir / 'model_bestG.pth')
            best_pathD = str(self.checkpoint_dir / 'model_bestD.pth')
            torch.save(stateG, best_pathG)
            torch.save(stateD, best_pathD)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = Path(resume_path)
        with open(resume_path / "checkpoint", "r") as f:
            lines = f.readlines()
            lines = list(map(str.strip, lines))
            if len(lines) != 2:
                self.logger.warning("Warning: Checkpoint is empty. Load model weight failed.")
                return
            resume_pathG, resume_pathD = lines

        self.logger.info("Loading checkpoint: {} and {} ...".format(resume_pathG, resume_pathD))

        checkpointG = torch.load(resume_pathG)
        checkpointD = torch.load(resume_pathD)
        self.start_epoch = checkpointG['epoch'] + 1
        # self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpointG['config']['netG'] != self.config['netG'] or checkpointD['config']['netD'] != self.config['netD']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.generator.load_state_dict(checkpointG['state_dict'])
        self.discriminator.load_state_dict(checkpointD['state_dict'])
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpointG['config']['optimizerG']['type'] != self.config['optimizerG']['type'] or \
                checkpointD['config']['optimizerD']['type'] != self.config['optimizerD']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizerG.load_state_dict(checkpointG['optimizer'])
            self.optimizerD.load_state_dict(checkpointD['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
