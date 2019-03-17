# -*- coding: utf-8 -*-
import os
import json
import platform

from pcode.utils.op_files import write_txt


class Logger:
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, file_folder):
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        self.file_json = os.path.join(file_folder, 'log.json')
        self.file_txt = os.path.join(file_folder, 'log.txt')
        self.values = []

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

    def log(self, value):
        print(value)
        self.save_txt(value)

    def save_json(self):
        """
        Save the internal memory to a file
        """
        with open(self.file_json, 'w') as fp:
            json.dump(self.values, fp, indent=' ')

    def save_txt(self, value):
        write_txt(value + "\n", self.file_txt, type="a")


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


def display_training_stat(args, scheduler, tracker):
    for name, stat in tracker.stat.items():
        args.logger.log_metric(
            name=name,
            values={
                'epoch': scheduler.epoch_,
                'local_index': scheduler.local_index, 'value': stat.avg},
            tags={'split': 'train'}
        )


def display_test_stat(args, scheduler, tracker, global_performance):
    for name, perf in zip(tracker.metrics_to_track, global_performance):
        args.logger.log_metric(
            name=name,
            values={
                'epoch': scheduler.epoch_, 'value': perf},
            tags={'split': 'test'}
        )
    args.logger.save_json()


def dispaly_best_test_stat(args, scheduler, best_tracker):
    args.logger.log(
        'best performance at local index {} \
        (best epoch {:.3f}, current epoch {:.3f}): {}.'.format(
            scheduler.local_index, best_tracker.get_best_perf_loc(),
            scheduler.epoch_, best_tracker.best_perf)
    )
