# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist

from pcode.utils.checkpoint import init_checkpoint
from pcode.utils.topology import FCGraph
from pcode.utils.logging import init_logging, display_args


def set_local_stat(args):
    args.local_index = 0
    args.best_primary_te_score = 0
    args.best_epoch = []


def init_config(args):
    # define the graph for the computation.
    cur_rank = dist.get_rank() if args.mpi_enabled else 0
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
    display_args(args)
