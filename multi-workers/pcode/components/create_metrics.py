# -*- coding: utf-8 -*-
from pcode.flow.communication import global_average
from pcode.tracking.meter import AverageMeter


def define_metrics(args, model):
    if 'least_square' not in args.arch:
        if model.num_classes >= 5:
            return (1, 5)
        else:
            return (1,)
    else:
        return ()


class TopKAccuracy(object):
    def __init__(self, topk=1):
        self.topk = topk
        self.reset()

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.size(0)

        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)

    def reset(self):
        self.top = AverageMeter()

    def update(self, prec, size):
        self.top.update(prec, size)

    def average(self):
        return global_average(self.top.sum, self.top.count)

    @property
    def name(self):
        return "Prec@{}".format(self.topk)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    res = []

    if len(topk) > 0:
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
    else:
        res += [0]
    return res
