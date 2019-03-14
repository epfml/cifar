# -*- coding: utf-8 -*-
import gc

import torch
import torch.distributed as dist

from pcode.components.create_scheduler import adjust_learning_rate
from pcode.components.create_dataset import \
    define_dataset, load_data_batch, _load_data_batch
from pcode.utils.checkpoint import save_to_checkpoint
from pcode.utils.logging import \
    display_training_stat, display_test_stat, dispaly_best_test_stat
from pcode.utils.meter import RuntimeTracker


def train_and_validate(args, model, criterion, scheduler, optimizer, metrics):
    """The training scheme of Hierarchical Local SGD."""
    print('start training and validation.')

    # get data loader.
    train_loader, val_loader = define_dataset(args, shuffle=True)

    # define runtime stat tracker and start the training.
    tracker_tr = RuntimeTracker(metrics_to_track=metrics.metric_names)

    # break until finish expected full epoch training.
    print('enter the training.')
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
            _input, _target = load_data_batch(args, _input, _target)

            # inference and get current performance.
            optimizer.zero_grad()
            loss = inference(model, criterion, metrics, _input, _target, tracker_tr)
            loss.backward()
            optimizer.step()

            # finish one epoch training and to decide if we want to val our model.
            if args.epoch_ % 1 == 0:
                # each worker finish one epoch training.
                do_validate(args, model, optimizer, criterion, metrics, val_loader)

                # refresh the logging cache at the begining of each epoch.
                tracker_tr.reset()

                # determine if the training is finished.
                if is_stop(args):
                    return

            # display the logging info.
            display_training_stat(args, tracker_tr)

        # reshuffle the data.
        if args.reshuffle_per_epoch:
            print('reshuffle the dataset.')
            del train_loader, val_loader
            gc.collect()
            print('reshuffle the dataset.')
            train_loader, val_loader = define_dataset(args, shuffle=True)


def inference(model, criterion, metrics, _input, _target, tracker):
    """Inference on the given model and get loss and accuracy."""
    output = model(_input)
    loss = criterion(output, _target)
    performance = metrics.evaluate(output, _target)
    tracker.update_metrics([loss] + performance, n_samples=_input.size(0))
    return loss


def do_validate(args, model, optimizer, criterion, metrics, val_loader):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function.
    dist.barrier()
    # evaluate the model.
    performance = validate(args, model, criterion, metrics, val_loader)

    # remember best prec@1 and save checkpoint.
    args.cur_primary_score = performance[0]
    is_best = args.cur_primary_te_score > args.best_primary_te_score
    if is_best:
        args.best_primary_te_score = performance[0]
        args.best_epoch += [args.epoch_]

    # logging and display val info.
    dispaly_best_test_stat(args)

    # save to the checkpoint.
    if args.graph.rank == 0:
        save_to_checkpoint({
            'current_epoch': args.epoch,
            'local_index': args.local_index,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': args.best_primary_te_score,
            },
            is_best, dirname=args.checkpoint_root,
            filename='checkpoint.pth.tar',
            save_all=args.save_all_models)
    print('Finished validation.')


def validate(args, model, criterion, metrics, val_loader):
    """A function for model evaluation."""
    # define stat.
    tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

    # switch to evaluation mode
    model.eval()

    print('Do validation.')
    for _input, _target in val_loader:
        # load data and check performance.
        _input, _target = _load_data_batch(args, _input, _target)

        with torch.no_grad():
            inference(model, criterion, metrics, _input, _target, tracker_te)

    # Aggregate and display the information.
    global_performance = tracker_te.evaluate_global_metrics()
    display_test_stat(args, tracker_te)
    return global_performance


"""some utility functions."""


def get_current_epoch(args):
    args.epoch_ = args.local_index / args.num_batches_train_per_device_per_epoch
    args.epoch = int(args.epoch_)


def is_stop(args):
    if args.stop_criteria == 'epoch':
        return args.epoch >= args.num_epochs
    elif args.stop_criteria == 'iteration':
        return args.local_index >= args.num_iterations_per_worker
