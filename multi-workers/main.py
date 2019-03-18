# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist

from parameters import get_args
from pcode.create_components import create_components
from pcode.distributed_running import train_and_validate
from pcode.utils.checkpoint import init_checkpoint
from pcode.utils.topology import define_graph_topology
from pcode.utils.logging import Logger, display_args


def main(conf):
    try:
        dist.init_process_group('mpi')
        conf.mpi_enabled = True
    except AttributeError as e:
        conf.mpi_enabled = False

    # init the config.
    init_config(conf)

    # create model and deploy the model.
    model, criterion, scheduler, optimizer, metrics = create_components(conf)

    # train amd evaluate model.
    train_and_validate(conf, model, criterion, scheduler, optimizer, metrics)


def init_config(conf):
    # define the graph for the computation.
    cur_rank = dist.get_rank() if conf.mpi_enabled else 0
    conf.graph = define_graph_topology(
        rank=cur_rank, n_nodes=conf.n_nodes,
        on_cuda=conf.on_cuda, world=conf.world,
        graph_topology='complete')

    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(conf.manual_seed)
        torch.cuda.set_device(conf.graph.device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # define checkpoint for logging.
    init_checkpoint(conf)

    # configure logger.
    conf.logger = Logger(conf.checkpoint_dir)

    # display the arguments' info.
    display_args(conf)


if __name__ == '__main__':
    conf = get_args()
    main(conf)
