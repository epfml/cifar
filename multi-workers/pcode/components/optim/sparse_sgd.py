# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

from pcode.utils.communication import get_aggregator_fn


class SparseSGD(Optimizer):
    r"""Implements sparsified/quantized version of stochastic gradient descent.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)

        """

    def __init__(self,
                 params,
                 lr=required,
                 dampening=0,
                 weight_decay=0,
                 momentum=0,
                 conf=None):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, dampening=dampening)

        self.world_size = len(conf.graph.ranks)
        self.sparsification_scheme = conf.sparsification_scheme
        self.communication_scheme = conf.communication_scheme
        self.num_coordinates = conf.num_coordinates

        super(SparseSGD, self).__init__(params, defaults)

        self.__create_gradients_memory()

        # store the whole training arguments.
        self.conf = conf

        # define the aggregator.
        self.aggregator = get_aggregator_fn(
            aggregator_name='centralized',
            rank=conf.graph.rank, neighbors=conf.graph.ranks)

        if self.communication_scheme == "all_reduce":
            self.rng = np.random.RandomState(100)
        else:
            self.rng = np.random.RandomState()

    def __create_gradients_memory(self):
        r""" Create a memory to keep gradients that are not used in each iteration """

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['memory'] = torch.zeros_like(p.data.view(-1))

    def sparsify_gradients(self, param, lr):
        """ Calls one of the sparsification functions (random or blockwise)

        Args:
            param (:obj: `torch.nn.Parameter`): Model parameter
            lr (float): Learning rate
        """
        if self.sparsification_scheme == "random":
            return self._random_sparsify(param, lr)
        else:
            raise NotImplementedError

    def _random_sparsify(self, param, lr):
        """ Sparsify the gradients vector by selecting 'k' of them randomly.

        Args:
            param (:obj: `torch.nn.Parameter`): Model parameter
            lr (float): Learning rate
        """
        grad_view = param.grad.data.view(-1)
        full_size = len(grad_view)

        # Update memory
        self.state[param]['memory'].add_(grad_view * lr)

        # Number of coordinates to use
        num_coordinates = self._sparse_vector_size(full_size)

        if num_coordinates < full_size:
            selected_indices = torch.tensor(
                self.rng.choice(full_size, num_coordinates, replace=False))
        else:
            selected_indices = torch.tensor(np.arange(full_size))

        # Create a sparse vector for the (index, value) of sampled weight
        sparse_tensor = self.state[param]['memory'][selected_indices]
        self.state[param]['memory'][selected_indices] -= sparse_tensor

        return sparse_tensor, selected_indices

    def _sparse_vector_size(self, full_size):
        """ return size of the sparse tensor """
        if self.num_coordinates < 1:
            num_coordinates = max(int(full_size * self.num_coordinates), 1)
        elif self.num_coordinates >= full_size:
            num_coordinates = full_size
        else:
            num_coordinates = self.num_coordinates

        return num_coordinates

    def step(self, closure=None):
        """ Sparsifies/Quantize gradients, aggregates the gradients, and performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for i, p in enumerate(group['params']):
                if weight_decay != 0:
                    p.grad.data.add_(weight_decay, p.data)

                # Make the gradients sparse
                sparse_tensor, indices = self.sparsify_gradients(p, lr)
                # Aggregate the gradients
                self._aggregate_sparsified_gradients(p, sparse_tensor, indices)

                if p.grad is None:
                    continue
                d_p = p.grad.data

                # apply the momentum.
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf = momentum * buf + d_p
                    d_p = buf
                p.data.add_(-d_p)

    def _aggregate_sparsified_gradients(self, param, sparse_tensor, indices):

        """ Aggregates the sparsified/quantized gradients """
        param.grad.data = torch.zeros_like(param.grad.data)
        dv = param.grad.data.view(-1)

        if self.communication_scheme == "all_gather":
            # gather gradients and indices
            gradients_list = self.aggregator._agg(
                sparse_tensor,
                mpi_enabled=self.conf.mpi_enabled,
                communication_scheme="all_gather"
            )
            indices_list = self.aggregator._agg(
                indices,
                mpi_enabled=self.conf.mpi_enabled,
                communication_scheme="all_gather"
            )

            # re-organize the gathered information.
            for grad, indices in zip(gradients_list, indices_list):
                dv[indices] += grad
            dv /= self.world_size
        elif self.communication_scheme == "all_reduce":
            self.aggregator._agg(
                sparse_tensor, op='avg',
                mpi_enabled=self.conf.mpi_enabled,
                communication_scheme="all_reduce"
            )

            dv[indices] = sparse_tensor
        else:
            raise NotImplementedError
