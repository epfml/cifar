# -*- coding: utf-8 -*-
import torch.nn as nn

from pcode.components.create_optimizer import define_optimizer
from pcode.components.create_metrics import Metrics
from pcode.components.create_model import define_model
from pcode.components.create_scheduler import LRScheduler
from pcode.utils.checkpoint import maybe_resume_from_checkpoint


def create_components(conf):
    """Create model, criterion and optimizer.
    If conf.use_cuda is True, use ps_id as GPU_id.
    """
    model = define_model(conf)

    # define the criterion and metrics.
    criterion = define_criterion(conf)
    metrics = Metrics(model)

    # place model and criterion.
    if conf.graph.on_cuda:
        model.cuda()
        criterion = criterion.cuda()

    # define the optimizer.
    optimizer = define_optimizer(conf, model)

    # (optional) reload checkpoint
    maybe_resume_from_checkpoint(conf, model, optimizer)

    # define the lr scheduler.
    scheduler = LRScheduler(conf, optimizer)
    return model, criterion, scheduler, optimizer, metrics


def define_criterion(conf):
    if 'least_square' in conf.arch:
        criterion = nn.MSELoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    return criterion
