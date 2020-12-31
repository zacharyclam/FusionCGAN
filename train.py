from __future__ import absolute_import

import argparse
import collections
import os

import torch

import data_loader.data_loaders as module_data
import model.discriminator as module_discriminator
import model.loss as module_loss
import model.metric as module_metric
import model.generator as module_generator
from parse_config import ConfigParser
from trainer.trainer import Trainer


# fix random seeds for reproducibility
random_seed = 1234
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

torch.backends.cudnn.enable = True  # 使用非确定性算法
torch.backends.cudnn.deterministic = True  # 保持CPU和GPU结果一致
torch.backends.cudnn.benchmark = True  # 提升计算速度


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_data_loader = config.init_obj('train_data_loader', module_data)
    test_data_loader = config.init_obj('test_data_loader', module_data)
    eval_data_loader = config.init_obj('eval_data_loader', module_data)

    # build model architecture, then print to console
    generator = config.init_obj('netG', module_generator)
    discriminator = config.init_obj('netD', module_discriminator)

    logger.info(generator)
    logger.info(discriminator)

    # get function handles of loss and metrics

    criterion = {loss: getattr(module_loss, loss) for loss in config['loss']}
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizerG = config.init_obj('optimizerG', torch.optim, generator.parameters())
    optimizerD = config.init_obj('optimizerD', torch.optim, discriminator.parameters())

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizerD)

    trainer = Trainer(generator, discriminator, criterion, metrics, optimizerG, optimizerD,
                      config=config,
                      data_loader=train_data_loader,
                      test_data_loader=test_data_loader,
                      eval_data_loader=eval_data_loader,
                      lr_scheduler=lr_scheduler
                      )
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='paremeter')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(args, options)


    main(config)
