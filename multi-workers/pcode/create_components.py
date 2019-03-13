# -*- coding: utf-8 -*-
import torch.nn as nn

from pcode.components.create_optimizer import define_optimizer
from pcode.components.create_metrics import define_metrics
from pcode.components.create_model import define_model
from pcode.components.create_scheduler import define_scheduler
from pcode.tracking.checkpoint import maybe_resume_from_checkpoint


def create_components(args):
    """Create model, criterion and optimizer.
    If args.use_cuda is True, use ps_id as GPU_id.
    """
    model = define_model(args)

    # define the criterion and metrics.
    criterion = define_criterion(args)
    metrics = define_metrics(args, model)

    # define the lr scheduler.
    scheduler = define_scheduler(args)

    # place model and criterion.
    if args.graph.on_cuda:
        model.cuda()
        criterion = criterion.cuda()

    # define the optimizer.
    optimizer = define_optimizer(args, model)

    # (optional) reload checkpoint
    maybe_resume_from_checkpoint(args, model, optimizer)
    return model, criterion, scheduler, optimizer, metrics


def define_criterion(args):
    if 'least_square' in args.arch:
        criterion = nn.MSELoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    return criterion
