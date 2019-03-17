# -*- coding: utf-8 -*-

import torch.distributed as dist

import pcode.components.models as models


def define_model(conf):
    if 'wideresnet' in conf.arch:
        model = models.__dict__['wideresnet'](conf)
    elif 'resnet' in conf.arch:
        model = models.__dict__['resnet'](conf)
    elif 'densenet' in conf.arch:
        model = models.__dict__['densenet'](conf)
    else:
        model = models.__dict__[conf.arch](conf)

    # get a consistent init model over the world.
    if conf.mpi_enabled:
        consistent_model(conf, model)

    # get the model stat info.
    get_model_stat(conf, model)
    return model


def get_model_stat(conf, model):
    print("=> creating model '{}. total params for process {}: {}M".format(
        conf.arch, conf.graph.rank,
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        ))


def consistent_model(conf, model):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    print('consistent model for process (rank {})'.format(conf.graph.rank))
    cur_rank = conf.graph.rank
    for param in model.parameters():
        param.data = param.data if cur_rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
