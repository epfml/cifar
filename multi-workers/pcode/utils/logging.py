# -*- coding: utf-8 -*-
import os
import json
import platform


class JSONLogger:
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, filename, auto_save=True):
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        self.filename = filename
        self.values = []
        self.auto_save = auto_save

        # Ensure the output directory exists
        directory = os.path.dirname(self.filename)
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

    def log_metric(self, name, values, tags):
        """
        Store a scalar metric

        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        self.values.append({
            'measurement': name,
            **values,
            **tags,
        })
        print("{name}: {values} ({tags})".format(
            name=name, values=values, tags=tags))
        if self.auto_save:
            self.save()

    def save(self):
        """
        Save the internal memory to a file
        """
        with open(self.filename, 'w') as fp:
            json.dump(self.values, fp, indent=' ')


def init_logging(args):
    args.log_metric = JSONLogger(args.checkpoint_dir).log_metric


def log_metric(name, values, tags):
    """
    Log timeseries data.
    Placeholder implementation.
    This function should be overwritten by any script that runs this as a module.
    """
    print("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))


def display_args(args):
    print('parameters: ')
    for arg in vars(args):
        print(str(arg) + '\t' + str(getattr(args, arg)))
    for name in ['n_nodes', 'world', 'rank',
                 'ranks_with_blocks', 'blocks_with_ranks',
                 'device', 'on_cuda', 'get_neighborhood']:
        print('{}: {}'.format(name, getattr(args.graph, name)))

    print('experiment platform:')
    print(
        'Rank {} with block {} on {} {}-{}'.format(
            args.graph.rank,
            args.graph.ranks_with_blocks[args.graph.rank],
            platform.node(),
            'GPU' if args.graph.on_cuda else 'CPU',
            args.graph.device
            )
        )


def display_training_stat(args, tracker):
    for name, stat in tracker.stat.items():
        args.log_metric(
            name=name,
            values={
                'epoch': args.epoch_,
                'local_index': args.local_index, 'value': stat.avg},
            tags={'split': 'train'}
        )


def display_test_stat(args, tracker, global_performance):
    for name, perf in zip(tracker.metrics_to_track, global_performance):
        args.log_metric(
            name=name,
            values={
                'epoch': args.epoch_,
                'local_index': args.local_index, 'value': perf},
            tags={'split': 'test'}
        )


def dispaly_best_test_stat(args):
    print('best accuracy for rank {} at local index {} \
        (best epoch {:.3f}, current epoch {:.3f}): {}.'.format(
        args.graph.rank, args.local_index,
        args.best_epoch[-1] if len(args.best_epoch) != 0 else '',
        args.epoch_, args.best_primary_te_score))
