# -*- coding: utf-8 -*-
from pcode.flow.communication import global_average


def define_local_training_tracker():
    return define_trackers(['losses', 'top1', 'top5'])


def define_val_tracker():
    return define_trackers(['losses', 'top1', 'top5'])


def define_trackers(names):
    return dict((name, AverageMeter()) for name in names)


def evaluate_gloabl_performance(meter):
    return global_average(meter.sum, meter.count)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
