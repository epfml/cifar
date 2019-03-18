# -*- coding: utf-8 -*-
from abc import abstractmethod, ABC
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix


class UndirectedGraph(ABC):
    @property
    @abstractmethod
    def n_nodes(self):
        pass

    @property
    @abstractmethod
    def n_edges(self):
        pass

    @property
    @abstractmethod
    def beta(self):
        """
        `1 - beta` is the `spectral gap` of the mixing matrix.
        """
        pass

    @property
    @abstractmethod
    def matrix(self):
        """
        Doubly stochastic mixing matrix of the graph.
        """
        pass

    @property
    @abstractmethod
    def world(self):
        pass

    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def ranks(self):
        pass

    @property
    @abstractmethod
    def device(self):
        pass

    @property
    @abstractmethod
    def on_cuda(self):
        pass

    @abstractmethod
    def get_neighborhood(self, node_id):
        pass


class PhysicalLayout(UndirectedGraph):
    def __init__(self, rank, n_nodes, on_cuda, world=None):
        self._rank = rank
        self._n_nodes = n_nodes
        self._on_cuda = on_cuda
        self._world = world

    @property
    def device(self):
        return self.world[self.rank]

    @property
    def on_cuda(self):
        return self._on_cuda

    @property
    def rank(self):
        return self._rank

    @property
    def ranks(self):
        return list(range(self.n_nodes))

    @property
    def world(self):
        assert self._world is not None
        self._world_list = self._world.split(',')
        assert self._n_nodes <= len(self._world_list)
        return [int(l) for l in self._world_list]


class CompleteGraph(PhysicalLayout):
    def __init__(self, rank, n_nodes, on_cuda, world=None):
        super(CompleteGraph, self).__init__(rank, n_nodes, on_cuda, world)
        self._data = np.ones((n_nodes, n_nodes)) / n_nodes

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_edges(self):
        return self._n_nodes * (self._n_nodes - 1) / 2

    @property
    def beta(self):
        return 0

    @property
    def matrix(self):
        return self._data

    def get_neighborhood(self):
        """it will return a list of ranks that are connected with this node."""
        row = self._data[self._rank]
        return {c: v for c, v in zip(range(len(row)), row)}


class RingGraph(PhysicalLayout):
    def __init__(self, rank, n_nodes, on_cuda, world=None):
        super(RingGraph, self).__init__(rank, n_nodes, on_cuda, world)
        self._data, self._beta = self._compute_beta(n_nodes)

    def _compute_beta(self, n):
        assert n > 2

        # create ring matrix
        diag_rows = np.array([[1/3 for _ in range(n)], [1/3 for _ in range(n)], [1/3 for _ in range(n)]])
        positions = [-1, 0, 1]
        data = sp.sparse.spdiags(diag_rows, positions, n, n).tolil()
        data[0, n-1] = 1/3
        data[n-1, 0] = 1/3
        data = data.tocsr()

        if n > 3:
            # Find largest real part
            eigenvalues, _ = eigs(data, k=2, which='LR')
            lambda2 = min(abs(i.real) for i in eigenvalues)

            # Find smallest real part
            eigenvalues, _ = eigs(data, k=1, which='SR')
            lambdan = eigenvalues[0].real
        else:
            eigenvals = sorted(np.linalg.eigvals(data.toarray()), reverse=True)
            lambda2 = eigenvals[1]
            lambdan = eigenvals[-1]

        beta = max(abs(lambda2), abs(lambdan))
        return data, beta

    @property
    def n_edges(self):
        return self._n_nodes

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def beta(self):
        return self._beta

    @property
    def matrix(self):
        return self._data

    def get_neighborhood(self):
        row = self._data.getrow(self._rank)
        _, cols = row.nonzero()
        vals = row.data
        return {int(c): v for c, v in zip(cols, vals)}


def define_graph_topology(rank, n_nodes, on_cuda, world, graph_topology,  **args):
    """Return the required graph object.
    Parameters
    ----------
    n_nodes : {int}
        Number of nodes in the network.
    graph_topology : {str}
        A string describing the graph topology
    Returns
    -------
    Graph
        A graph with specified information.
    """
    if graph_topology == 'ring':
        graph = RingGraph(rank, n_nodes, on_cuda, world)
    elif graph_topology == 'complete':
        graph = CompleteGraph(rank, n_nodes, on_cuda, world)
    return graph