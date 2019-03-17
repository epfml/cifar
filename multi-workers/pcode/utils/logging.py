# -*- coding: utf-8 -*-
import os
import json
import time
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
        content = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + value
        print(content)
        self.save_txt(content)

    def save_json(self):
        """
        Save the internal memory to a file
        """
        with open(self.file_json, 'w') as fp:
            json.dump(self.values, fp, indent=' ')

    def save_txt(self, value):
        write_txt(value + "\n", self.file_txt, type="a")


def display_args(conf):
    print('\n\nparameters: ')
    for arg in vars(conf):
        print('\t' + str(arg) + '\t' + str(getattr(conf, arg)))

    print('\n\nexperiment platform: rank {} with block {} on {} {}-{}'.format(
            conf.graph.rank,
            conf.graph.ranks_with_blocks[conf.graph.rank],
            platform.node(),
            'GPU' if conf.graph.on_cuda else 'CPU',
            conf.graph.device
            ))
    for name in ['n_nodes', 'world', 'rank',
                 'ranks_with_blocks', 'blocks_with_ranks',
                 'device', 'on_cuda', 'get_neighborhood']:
        print('\t{}: {}'.format(name, getattr(conf.graph, name)))
    print('\n\n')


def display_training_stat(conf, scheduler, tracker):
    for name, stat in tracker.stat.items():
        conf.logger.log_metric(
            name=name,
            values={
                'rank': conf.graph.rank,
                'epoch': scheduler.epoch_,
                'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'local_index': scheduler.local_index, 'value': stat.avg},
            tags={'split': 'train'}
        )


def display_test_stat(conf, scheduler, tracker, global_performance):
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S")

    for name, perf in zip(tracker.metrics_to_track, global_performance):
        conf.logger.log_metric(
            name=name,
            values={
                'rank': conf.graph.rank,
                'epoch': scheduler.epoch_,
                'time': cur_time,
                'value': perf},
            tags={'split': 'test'}
        )
    conf.logger.save_json()


def dispaly_best_test_stat(conf, scheduler, best_tracker):
    conf.logger.log(
        'best performance at local index {} \
        (best epoch {:.3f}, current epoch {:.3f}): {}.'.format(
            scheduler.local_index, best_tracker.get_best_perf_loc(),
            scheduler.epoch_, best_tracker.best_perf)
    )
