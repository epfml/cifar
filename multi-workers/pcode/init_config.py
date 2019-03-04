# -*- coding: utf-8 -*-
import platform

import torch
import torch.distributed as dist

from pcode.tracking.checkpoint import init_checkpoint
from pcode.utils.topology import FCGraph
from pcode.tracking.logging import init_logging, log_args, info


def set_local_stat(args):
    args.local_index = 0
    args.best_prec1 = 0
    args.best_epoch = []
    args.tracking = []


def init_config(args):
    # define the graph for the computation.
    cur_rank = dist.get_rank()
    args.graph = FCGraph(cur_rank, args.blocks, args.on_cuda, args.world)

    if args.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.set_device(args.graph.device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # local conf.
    set_local_stat(args)

    # define checkpoint for logging.
    init_checkpoint(args)

    # define the logging scheme.
    init_logging(args)

    # display the arguments' info.
    log_args(args)
    info(
        'Rank {} with block {} on {} {}-{}'.format(
            args.graph.rank,
            args.graph.ranks_with_blocks[args.graph.rank],
            platform.node(),
            'GPU' if args.graph.on_cuda else 'CPU',
            args.graph.device
            )
        )
