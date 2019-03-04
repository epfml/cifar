# -*- coding: utf-8 -*-
import gc

import torch
import torch.distributed as dist

from pcode.components.create_metrics import accuracy
from pcode.components.create_scheduler import adjust_learning_rate
from pcode.components.create_dataset import define_dataset, load_data_batch, \
    _load_data_batch
from pcode.tracking.checkpoint import save_to_checkpoint
from pcode.flow.flow_utils import is_stop, get_current_epoch
from pcode.flow.communication import aggregate_gradients
from pcode.tracking.logging import info, \
    logging_display_training, logging_display_val, \
    update_performance_tracker
from pcode.tracking.meter import define_local_training_tracker,\
    define_val_tracker, evaluate_gloabl_performance


def train_and_validate(args, model, criterion, scheduler, optimizer, metrics):
    """The training scheme of Hierarchical Local SGD."""
    info('start training and validation.')

    # get data loader.
    train_loader, val_loader = define_dataset(args, shuffle=True)

    if args.evaluate:
        validate(args, model, criterion, metrics, val_loader)
        return

    # init global variable.
    tracker = define_local_training_tracker()
    info('enter the training.')

    # break until finish expected full epoch training.
    while True:
        # configure local step.
        for _input, _target in train_loader:
            model.train()

            # update local index and get local step
            args.local_index += 1
            get_current_epoch(args)

            # adjust learning rate (based on the # of accessed samples)
            adjust_learning_rate(args, optimizer, scheduler)

            # load data
            _input, _target = load_data_batch(args, _input, _target, tracker)

            # inference and get current performance.
            optimizer.zero_grad()
            loss, performance = inference(model, criterion, metrics, _input, _target)
            loss.backward()

            # update performance tracker
            update_performance_tracker(tracker, loss, performance, _input.size(0))

            # sync and broadcast gradients to other nodes by using reduce_sum.
            aggregate_gradients(args, model)
            optimizer.step()

            # finish one epoch training and to decide if we want to val our model.
            if args.epoch_ % 1 == 0:
                # each worker finish one epoch training.
                do_validate(args, model, optimizer, criterion, metrics, val_loader)

                # refresh the logging cache at the begining of each epoch.
                tracker = define_local_training_tracker()

                # determine if the training is finished.
                if is_stop(args):
                    return

            # display the logging info.
            logging_display_training(args, tracker)

        # reshuffle the data.
        if args.reshuffle_per_epoch:
            info('reshuffle the dataset.')
            del train_loader, val_loader
            gc.collect()
            info('reshuffle the dataset.')
            train_loader, val_loader = define_dataset(args, shuffle=True)


def inference(model, criterion, metrics, _input, _target):
    """Inference on the given model and get loss and accuracy."""
    output = model(_input)
    loss = criterion(output, _target)
    performance = accuracy(output.data, _target, topk=metrics)
    return loss, performance


def do_validate(args, model, optimizer, criterion, metrics, val_loader):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function.
    dist.barrier()
    # evaluate the model.
    performance = validate(args, model, criterion, metrics, val_loader)

    # remember best prec@1 and save checkpoint.
    args.cur_prec1 = performance[0]
    is_best = args.cur_prec1 > args.best_prec1
    if is_best:
        args.best_prec1 = performance[0]
        args.best_epoch += [args.epoch_]

    # logging and display val info.
    logging_display_val(args)

    # save to the checkpoint.
    if args.graph.rank == 0:
        save_to_checkpoint({
            'arguments': args,
            'current_epoch': args.epoch,
            'local_index': args.local_index,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': args.best_prec1,
            },
            is_best, dirname=args.checkpoint_root,
            filename='checkpoint.pth.tar',
            save_all=args.save_all_models)
    info('finished validation.')


def validate(args, model, criterion, metrics, val_loader):
    """A function for model evaluation."""
    # define stat.
    tracker = define_val_tracker()

    # switch to evaluation mode
    model.eval()

    info('Do validation.')
    for _input, _target in val_loader:
        # load data and check performance.
        _input, _target = _load_data_batch(args, _input, _target)

        with torch.no_grad():
            loss, performance = inference(
                model, criterion, metrics, _input, _target)
            tracker = update_performance_tracker(
                tracker, loss, performance, _input.size(0))

    info('Aggregate val accuracy from different partitions.')
    performance = [
        evaluate_gloabl_performance(tracker[x]) for x in ['top1', 'top5']
    ]

    info('Val at batch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f}'.format(
        args.local_index, args.graph.rank, performance[0], performance[1]))
    return performance
