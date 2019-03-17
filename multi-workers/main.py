# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist

from parameters import get_args
from pcode.create_components import create_components
from pcode.distributed_running import train_and_validate
from pcode.utils.checkpoint import init_checkpoint
from pcode.utils.topology import FCGraph
from pcode.utils.logging import Logger, display_args


def main(args):
    try:
        dist.init_process_group('mpi')
        args.mpi_enabled = True
    except AttributeError as e:
        args.mpi_enabled = False

    # init the config.
    init_config(args)

    # create model and deploy the model.
    model, criterion, scheduler, optimizer, metrics = create_components(args)

    # train amd evaluate model.
    train_and_validate(args, model, criterion, scheduler, optimizer, metrics)


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

    # define checkpoint for logging.
    init_checkpoint(args)

    # configure logger.
    args.logger = Logger(args.checkpoint_dir)

    # display the arguments' info.
    display_args(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
