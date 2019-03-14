# -*- coding: utf-8 -*-
from copy import deepcopy

import torch

from pcode.components.optim.utils.communication import global_average


class Mean(object):
    """
    Running average of the values that are 'add'ed
    """
    def __init__(self, update_weight=1):
        """
        :param update_weight: 1 for normal, 2 for t-average
        """
        self.average = None
        self.counter = 0
        self.update_weight = update_weight

    def add(self, value, weight=1):
        """Add a value to the accumulator"""
        self.counter += weight
        if self.average is None:
            self.average = deepcopy(value)
        else:
            delta = value - self.average
            self.average += delta * self.update_weight * weight / (self.counter + self.update_weight - 1)
            if isinstance(self.average, torch.Tensor):
                self.average.detach()

    def value(self):
        """Access the current running average"""
        return self.average


class Max(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """
    def __init__(self):
        self.max = None

    def add(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.max


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float('inf')
        self.min = float('inf')
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min


class RuntimeTracker(object):
    """Tracking the runtime stat for local training."""
    def __init__(self, metrics_to_track=['top1']):
        self.metrics_to_track = metrics_to_track
        self.things_to_track = ['loss'] + metrics_to_track
        self.reset()

    def reset(self):
        self.stat = dict(
            (name, AverageMeter()) for name in self.things_to_track)

    def evaluate_global_metric(self, metric):
        return global_average(self.stat[metric].sum, self.stat[metric].count)

    def evaluate_global_metrics(self):
        return [self.evaluate_global_metric(metric) for metric in self.metrics_to_track]

    def update_metrics(self, metric_stat, n_samples):
        for idx, thing in enumerate(self.things_to_track):
            self.stat[thing].update(metric_stat[idx], n_samples)
