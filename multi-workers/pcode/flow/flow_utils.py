# -*- coding: utf-8 -*-

"""some utils function to control the flow."""


def get_current_epoch(args):
    args.epoch_ = args.local_index / args.num_batches_train_per_device_per_epoch
    args.epoch = int(args.epoch_)


def is_stop(args):
    if args.stop_criteria == 'epoch':
        return args.epoch >= args.num_epochs
    elif args.stop_criteria == 'iteration':
        return args.local_index >= args.num_iterations_per_worker
