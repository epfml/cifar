# -*- coding: utf-8 -*-
import gc
import shutil
import time
from os.path import join, isfile

import torch

from pcode.utils.op_paths import build_dirs


def get_checkpoint_folder_name(conf):
    # return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    time_id = str(int(time.time()))
    time_id += '_l2-{}_lr-{}_epochs-{}_batchsize-{}_num_nodes_{}_optim-{}'.format(
        conf.weight_decay,
        conf.lr,
        conf.num_epochs,
        conf.batch_size,
        conf.n_nodes,
        conf.optimizer
    )
    return time_id


def init_checkpoint(conf):
    # init checkpoint dir.
    conf.checkpoint_root = join(
        conf.checkpoint, conf.data, conf.arch,
        conf.experiment if conf.experiment is not None else '',
        conf.timestamp)
    conf.checkpoint_dir = join(conf.checkpoint_root, str(conf.graph.rank))
    if conf.save_some_models is not None:
        conf.save_some_models = conf.save_some_models.split(',')

    # if the directory does not exists, create them.
    build_dirs(conf.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_to_checkpoint(
        conf, state, is_best, dirname, filename, save_all=False):
    # save full state.
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(checkpoint_path, join(
            dirname,
            'checkpoint_epoch_%s.pth.tar' % state['current_epoch']))
    elif conf.save_some_models is not None:
        if str(state['current_epoch']) in conf.save_some_models:
            shutil.copyfile(checkpoint_path, join(
                dirname,
                'checkpoint_epoch_%s.pth.tar' % state['current_epoch'])
            )


def maybe_resume_from_checkpoint(conf, model, optimizer):
    if conf.resume:
        if conf.checkpoint_index is not None:
            # reload model from a specific checkpoint index.
            checkpoint_index = '_epoch_' + conf.checkpoint_index
        else:
            # reload model from the latest checkpoint.
            checkpoint_index = ''
        checkpoint_path = join(
            conf.resume, 'checkpoint{}.pth.tar'.format(checkpoint_index))
        print('try to load previous model from the path:{}'.format(
              checkpoint_path))

        if isfile(checkpoint_path):
            print("=> loading checkpoint {} for {}".format(
                conf.resume, conf.graph.rank))

            # get checkpoint.
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # restore some run-time info.
            conf.local_index = checkpoint['local_index']
            conf.best_perf = checkpoint['best_perf']

            # reset path for log.
            # remove_folder(conf.checkpoint_root)
            conf.checkpoint_root = conf.resume
            conf.checkpoint_dir = join(conf.resume, str(conf.graph.rank))
            # restore model.
            model.load_state_dict(checkpoint['state_dict'])
            # restore optimizer.
            optimizer.load_state_dict(checkpoint['optimizer'])
            # logging.
            print("=> loaded model from path '{}' checkpointed at (epoch {})"
                  . format(conf.resume, checkpoint['current_epoch']))

            # try to solve memory issue.
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
            return
        else:
            print("=> no checkpoint found at '{}'".format(conf.resume))
