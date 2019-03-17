# -*- coding: utf-8 -*-
import gc

import torch
import torch.distributed as dist

from pcode.components.create_scheduler import adjust_learning_rate
from pcode.components.create_dataset import define_dataset, load_data_batch
from pcode.utils.checkpoint import save_to_checkpoint
from pcode.utils.logging import \
    display_training_stat, display_test_stat, dispaly_best_test_stat
from pcode.utils.stat_tracker import RuntimeTracker, BestPerf


def train_and_validate(args, model, criterion, scheduler, optimizer, metrics):
    """The training scheme of Hierarchical Local SGD."""
    print('start training and validation.')

    # get data loader.
    train_loader, val_loader = define_dataset(args, shuffle=True)

    # define runtime stat tracker and start the training.
    tracker_tr = RuntimeTracker(metrics_to_track=metrics.metric_names)
    best_tracker = BestPerf(best_perf=None if 'best_perf' not in args else args.best_perf)

    # break until finish expected full epoch training.
    print('enter the training.')
    while True:
        # configure local step.
        for _input, _target in train_loader:
            model.train()

            # load data
            _input, _target = load_data_batch(args, _input, _target)

            # inference and get current performance.
            scheduler.step()
            optimizer.zero_grad()
            loss = inference(model, criterion, metrics, _input, _target, tracker_tr)
            loss.backward()
            optimizer.step()

            # finish one epoch training and to decide if we want to val our model.
            if args.epoch_ % 1 == 0:
                # each worker finish one epoch training.
                do_validate(
                    args, model, optimizer, criterion, scheduler, metrics,
                    val_loader, best_tracker)

                # refresh the logging cache at the begining of each epoch.
                tracker_tr.reset()

                # determine if the training is finished.
                if scheduler.is_stop():
                    return

            # display the logging info.
            display_training_stat(args, scheduler, tracker_tr)

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


def do_validate(
        args, model, optimizer, criterion, scheduler, metrics,
        val_loader, best_tracker):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    dist.barrier()
    performance = validate(
        args, model, criterion, scheduler, metrics, val_loader)

    # remember best performance and display the val info.
    best_tracker.update(performance[0], scheduler.epoch_)
    dispaly_best_test_stat(args, scheduler, best_tracker)

    # save to the checkpoint.
    if args.graph.rank == 0:
        save_to_checkpoint({
            'current_epoch': scheduler.epoch,
            'local_index': scheduler.local_index,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_tracker.best_perf,
            },
            best_tracker.is_best, dirname=args.checkpoint_root,
            filename='checkpoint.pth.tar',
            save_all=args.save_all_models)
    print('Finished validation.')


def validate(args, model, criterion, scheduler, metrics, val_loader):
    """A function for model evaluation."""
    # define stat.
    tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

    # switch to evaluation mode
    model.eval()

    print('Do validation.')
    for _input, _target in val_loader:
        # load data and check performance.
        _input, _target = load_data_batch(args, _input, _target)

        with torch.no_grad():
            inference(model, criterion, metrics, _input, _target, tracker_te)

    # Aggregate and display the information.
    global_performance = tracker_te.evaluate_global_metrics()
    display_test_stat(args, scheduler, tracker_te, global_performance)
    return global_performance
