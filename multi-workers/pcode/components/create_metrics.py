# -*- coding: utf-8 -*-
from pcode.tracking.meter import AverageMeter


def define_metrics(args, model):
    if 'least_square' not in args.arch:
        if model.num_classes >= 5:
            return (1, 5)
        else:
            return (1,)
    else:
        return ()


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
