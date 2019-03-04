# -*- coding: utf-8 -*-
import torch.distributed as dist

from parameters import get_args
from pcode.components.create_components import create_components
from pcode.init_config import init_config
from pcode.flow.distributed_running import train_and_validate


def main(args):
    """distributed training via mpi backend."""
    dist.init_process_group('mpi')

    # init the config.
    init_config(args)

    # create model and deploy the model.
    model, criterion, scheduler, optimizer, metrics = create_components(args)

    # train amd evaluate model.
    train_and_validate(args, model, criterion, scheduler, optimizer, metrics)


if __name__ == '__main__':
    args = get_args()
    main(args)
