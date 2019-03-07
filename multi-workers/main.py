# -*- coding: utf-8 -*-
import torch.distributed as dist

from parameters import get_args
from pcode.components.create_components import create_components
from pcode.init_config import init_config
from pcode.flow.distributed_running import train_and_validate


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



if __name__ == '__main__':
    args = get_args()
    main(args)
