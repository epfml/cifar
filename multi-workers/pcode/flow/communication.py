# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist


"""aggregate gradient"""


def aggregate_gradients(args, model):
    """Aggregate gradients.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.

    The previous synchronized model could be either from block/global sync,
    and the all-reduce range (group), can also be determined by sync status.

    We have a flag, i.e., args.avg_model, to determine if we want to average
    these gradients/difference or simply sum them up.
    """
    for param in model.parameters():
        # all reduce.
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        # if or not averge the model.
        if args.avg_model:
            param.grad.data /= float(args.graph.n_nodes)


"""functions."""


def global_average(sum, count):
    def helper(array):
        array = torch.FloatTensor(array)
        dist.all_reduce(array, op=dist.ReduceOp.SUM)
        all_sum, all_count = array
        if all_count == 0:
            return 0
        else:
            return all_sum / all_count
    avg = helper([sum, count])
    return avg


def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor
