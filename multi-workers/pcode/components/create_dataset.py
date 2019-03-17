# -*- coding: utf-8 -*-
import torch

from pcode.components.datasets.partition_data import DataPartitioner
from pcode.components.datasets.prepare_data import get_dataset


def load_data_batch(args, _input, _target):
    """Load a mini-batch and record the loading time."""
    if args.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()
    return _input, _target


def define_dataset(args, shuffle):
    print('create {} dataset for rank {}'.format(args.data, args.graph.rank))
    train_loader = partition_dataset(args, shuffle, dataset_type='train')
    val_loader = partition_dataset(args, shuffle, dataset_type='test')

    get_data_stat(args, train_loader, val_loader)
    return train_loader, val_loader


def partition_dataset(args, shuffle, dataset_type='train'):
    """ Given a dataset, partition it. """
    dataset = get_dataset(args, args.data, args.data_dir, split=dataset_type)
    batch_size = args.batch_size
    world_size = args.graph.n_nodes

    # partition data.
    if (args.partition_data or dataset_type == 'test') and args.mpi_enabled:
        partition_sizes = [1.0 / world_size for _ in range(world_size)]
        partition = DataPartitioner(args, dataset, shuffle, partition_sizes)
        data_to_load = partition.use(args.graph.rank)
        print('partitioned data and use subdata.')
    else:
        data_to_load = dataset
        print('used whole data.')

    if dataset_type == 'train':
        args.train_dataset_size = len(dataset)
        print('  We have {} samples for {}, \
             load {} data for process (rank {}).'.format(
             len(dataset), dataset_type, len(data_to_load), args.graph.rank))
    else:
        args.val_dataset_size = len(dataset)
        print('  We have {} samples for {}, \
             load {} val data for process (rank {}).'.format(
             len(dataset), dataset_type, len(data_to_load), args.graph.rank))

    # use Dataloader.
    data_type_label = (dataset_type == 'train')
    data_loader = torch.utils.data.DataLoader(
        data_to_load, batch_size=batch_size,
        shuffle=data_type_label,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        drop_last=False)

    print('we have {} batches for {} for rank {}.'.format(
         len(data_loader), dataset_type, args.graph.rank))
    return data_loader


def get_data_stat(args, train_loader, val_loader):
    # get the data statictics (on behalf of each worker) for train.
    args.num_batches_train_per_device_per_epoch = \
        len(train_loader) // args.graph.n_nodes \
        if not args.partition_data else len(train_loader)
    args.num_whole_train_batches_per_worker = \
        args.num_batches_train_per_device_per_epoch * args.num_epochs
    args.num_warmup_train_batches_per_worker = \
        args.num_batches_train_per_device_per_epoch * args.lr_warmup_epochs
    args.num_iterations_per_worker = args.num_iterations // args.graph.n_nodes

    # get the data statictics (on behalf of each worker) for val.
    args.num_batches_val_per_device_per_epoch = len(val_loader)

    # define some parameters for training.
    print('we have {} epochs, \
         {} mini-batches per device for training. \
         {} mini-batches per device for val. \
         The batch size: {}.'.format(
            args.num_epochs,
            args.num_batches_train_per_device_per_epoch,
            args.num_batches_val_per_device_per_epoch,
            args.batch_size))
