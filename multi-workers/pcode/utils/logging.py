# -*- coding: utf-8 -*-
import os
import json


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


def display_training_stat(args, tracker):
    log_info = 'Epoch: {epoch:.3f}. Local index: {local_index}. Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'.format(
        epoch=args.epoch_,
        local_index=args.local_index,
        loss=tracker['losses'].avg,
        top1=tracker['top1'].avg,
        top5=tracker['top5'].avg)
    print(log_info)


def display_test_stat(args):
    print('best accuracy for rank {} at local index {} \
        (best epoch {:.3f}, current epoch {:.3f}): {}.'.format(
        args.graph.rank, args.local_index,
        args.best_epoch[-1] if len(args.best_epoch) != 0 else '',
        args.epoch_, args.best_prec1))
