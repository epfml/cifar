# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist


"""some utils function to control the flow."""

def get_current_epoch(args):
    args.epoch_ = args.local_index / args.num_batches_train_per_device_per_epoch
    args.epoch = int(args.epoch_)


def is_stop(args):
    if args.stop_criteria == 'epoch':
        return args.epoch >= args.num_epochs
    elif args.stop_criteria == 'iteration':
        return args.local_index >= args.num_iterations_per_worker


"""some auxiliary functions for communication."""


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


def broadcast(tensor, src):
    return dist.broadcast(tensor, src=src)


"""some aggregation functions."""


class Aggregation(object):
    """Aggregate udpates / models from different processes."""

    def __init__(self, rank, neighbors):
        self.rank = rank
        self.neighbors = neighbors

        # get the world size from the view of the current rank.
        self.world_size = float(len(neighbors))

    def _agg(self, data, op):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        raise NotImplementedError

    def agg_model(self, model, op):
        """Aggregate models by model weight.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.data, op=op)
            param.data = grad

    def agg_grad(self, model, op):
        """Aggregate models gradients.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.grad.data, op=op)
            param.grad.data = grad


class CentralizedAggregation(Aggregation):
    """Aggregate udpates / models from different processes."""

    def __init__(self, rank, neighbors):
        super(CentralizedAggregation, self).__init__(rank, neighbors)
        assert rank in neighbors

    def _agg(self, data, op):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        if op == 'avg':
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            data /= self.world_size
        elif op == 'sum':
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
        else:
            raise NotImplementedError
        return data


class DecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, rank, neighbors):
        super(DecentralizedAggregation, self).__init__(rank, neighbors)
        assert rank not in neighbors

    def _agg(self, data, op):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        # Create some tensors to host the values from neighborhood.
        local_data = {i: torch.zeros_like(data) for i in self.neighbors}
        local_data[self.rank] = data

        reqs = []
        for node in self.neighbors:
            reqs.append(dist.isend(tensor=local_data[self.rank], dst=node))
            reqs.append(dist.irecv(tensor=local_data[node], src=node))

        for req in reqs:
            req.wait()

        # Aggregate local_data
        if op == 'avg':
            output = sum(local_data.values()) / (self.world_size + 1)
        else:
            raise NotImplementedError("op {} is not supported yet.".format(op))
        return output


def get_aggregator_fn(aggregator_name, rank, neighbors):
    return {
        'centralized': CentralizedAggregation,
        'decentralized': DecentralizedAggregation
    }[aggregator_name](rank, neighbors)
