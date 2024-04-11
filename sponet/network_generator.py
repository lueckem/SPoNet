from typing import Protocol
import networkx as nx
import numpy as np
from numpy.random import Generator, default_rng
import time


class NetworkGenerator(Protocol):
    num_agents: int

    def __call__(self) -> nx.Graph:
        """Generate a network."""

    def __repr__(self) -> str:
        """Return string representation of the generator."""

    def abrv(self) -> str:
        """Return short description for file names."""


class ErdosRenyiGenerator:
    def __init__(
        self,
        num_agents: int,
        p: float,
        max_sample_time: float = 10,
        rng: Generator = default_rng(),
        force_no_isolates: bool = False,
    ):
        """
        Generate Erdös-Renyi (binomial) random graphs.

        The random graph may contain isolated vertices, which is not allowed.
        In that case, the Generator samples until a valid network is found,
        or until max_sample_time seconds pass, in which case a RuntimeError is raised.

        Parameters
        ----------
        num_agents : int
        p : float
        max_sample_time : float, optional
            In seconds.
        rng : Generator, optional
            random number generator
        force_no_isolates : bool, optional
            If set to true, one random edge will be added to each isolated vertex,
            resulting in a network without isolates.
        """
        self.num_agents = num_agents
        self.p = p
        self.max_sample_time = max_sample_time
        self.rng = rng
        self.force_no_isolates = force_no_isolates

    def __call__(self) -> nx.Graph:
        gnp_fun = nx.erdos_renyi_graph if self.p > 0.2 else nx.fast_gnp_random_graph
        start = time.time()
        while True:
            network = gnp_fun(self.num_agents, self.p, seed=self.rng)
            if nx.number_of_isolates(network) == 0:
                return network

            if self.force_no_isolates:
                _unisolate_vertices(network)
                return network

            if time.time() - start > self.max_sample_time:
                raise RuntimeError(
                    "Timeout. Could not generate a graph without isolated vertices."
                )

    def __repr__(self) -> str:
        return f"Erdos-Renyi random graph with p={self.p} on {self.num_agents} nodes"

    def abrv(self) -> str:
        return f"ER_p{int(self.p * 100)}_N{self.num_agents}"


class RandomRegularGenerator:
    def __init__(self, num_agents: int, d: int, rng: Generator = default_rng()):
        """
        Generate random regular graphs.

        Parameters
        ----------
        num_agents : int
        d : int
        rng : Generator, optional
            random number generator
        """
        self.num_agents = num_agents
        self.d = d
        self.rng = rng

    def __call__(self) -> nx.Graph:
        return nx.random_regular_graph(self.d, self.num_agents, seed=self.rng)

    def __repr__(self) -> str:
        return (
            f"Uniform random regular graph with d={self.d} on {self.num_agents} nodes"
        )

    def abrv(self) -> str:
        return f"regular_d{self.d}_N{self.num_agents}"


class BarabasiAlbertGenerator:
    def __init__(self, num_agents: int, m: int, rng: Generator = default_rng()):
        """
        Generate random scale-free graphs using the Barabasi-Albert model.

        Parameters
        ----------
        num_agents : int
        m : int
        rng : Generator, optional
            random number generator
        """
        self.num_agents = num_agents
        self.m = m
        self.rng = rng

    def __call__(self) -> nx.Graph:
        return nx.barabasi_albert_graph(self.num_agents, self.m, seed=self.rng)

    def __repr__(self) -> str:
        return f"Barabasi-Albert random graph on {self.num_agents} nodes"

    def abrv(self):
        return f"barabasi_m{self.m}_N{self.num_agents}"


class WattsStrogatzGenerator:
    def __init__(
        self,
        num_agents: int,
        num_neighbors: int,
        p: float,
        rng: Generator = default_rng(),
    ):
        """
        Create random small-world networks using the Watts-Strogatz model.

        Parameters
        ----------
        num_agents : int
        num_neighbors : int
        p : float
        rng : Generator, optional
            random number generator
        """
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.p = p
        self.rng = rng

    def __call__(self) -> nx.Graph:
        return nx.connected_watts_strogatz_graph(
            self.num_agents, self.num_neighbors, self.p, seed=self.rng
        )

    def __repr__(self) -> str:
        return f"Watts-Strogatz random graph on {self.num_agents} nodes"

    def abrv(self):
        return f"watts_k{self.num_neighbors}_p{int(self.p * 100)}_N{self.num_agents}"


class StochasticBlockGenerator:
    def __init__(
        self,
        num_agents: int,
        p_matrix: np.ndarray,
        max_sample_time: float = 10,
        rng: Generator = default_rng(),
    ):
        """
        Creates n stochastic blocks, block i is randomly connected to block j with edge density p_matrix[i, j].

        The random graph may contain isolated vertices, which is not allowed.
        In that case, the Generator samples until a valid network is found,
        or until max_sample_time seconds pass, in which case a RuntimeError is raised.

        Parameters
        ----------
        num_agents : int
        p_matrix : np.ndarray
            (n x n) matrix of edge probabilities.
        max_sample_time : float, optional
            In seconds.
        rng : Generator, optional
            random number generator
        """
        self.p_matrix = p_matrix
        self.max_sample_time = max_sample_time

        self.num_blocks = p_matrix.shape[0]
        self.block_size = int(num_agents / self.num_blocks)
        self.num_agents = self.block_size * self.num_blocks
        self.rng = rng

    def _sample_adj_matrix(self):
        adj_matrix = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_blocks):
            for j in range(i + 1):
                this_block = (
                    self.rng.random((self.block_size, self.block_size))
                    <= self.p_matrix[i, j]
                )
                adj_matrix[
                    i * self.block_size : (i + 1) * self.block_size,
                    j * self.block_size : (j + 1) * self.block_size,
                ] = this_block

        # only keep the lower triangle and symmetrize
        adj_matrix = np.tril(adj_matrix, -1)
        adj_matrix = adj_matrix + np.tril(adj_matrix).T
        return adj_matrix

    def __call__(self) -> nx.Graph:
        start = time.time()
        while True:
            adj_mat = self._sample_adj_matrix()
            network = nx.from_numpy_array(adj_mat)
            if nx.number_of_isolates(network) == 0:
                return network

            if time.time() - start > self.max_sample_time:
                raise RuntimeError(
                    "Timeout. Could not generate a graph without isolated vertices."
                )

    def __repr__(self) -> str:
        return f"stochastic block model with {self.num_blocks} blocks and {self.num_agents} nodes."

    def abrv(self):
        return f"sbm_b{self.num_blocks}_N{self.num_agents}"


class GridGenerator:
    def __init__(self, num_agents: int, periodic: bool = False):
        """
        Generate lattice graph in 2 dimensions.

        Parameters
        ----------
        num_agents : int
        periodic : bool, optional
        """
        self.num_agents = num_agents
        self.dim = 2  # only implemented for 2D grids for now
        self.periodic = periodic

        # find shape as square as possible
        shape0 = round(num_agents**0.5)
        while num_agents % shape0 != 0:
            shape0 += 1
        shape1 = int(num_agents / shape0)
        self.shape = shape0, shape1

    def __call__(self) -> nx.Graph:
        g = nx.grid_graph(self.shape, periodic=self.periodic)

        relabel_dict = {}
        for i, val in enumerate(g.nodes):
            relabel_dict[val] = i

        return nx.relabel_nodes(g, relabel_dict)

    def __repr__(self) -> str:
        return f"{self.shape} lattice graph graph"

    def abrv(self):
        if self.periodic:
            return f"lattice_{self.dim}D_periodic_N{self.num_agents}"
        else:
            return f"lattice_{self.dim}D_N{self.num_agents}"


class BinomialWattsStrogatzGenerator:
    def __init__(
        self,
        num_agents: int,
        num_neighbors: int,
        p_rewire: float,
        max_sample_time: float = 10,
        rng: Generator = default_rng(),
    ):
        """
        Creates a ring where each node is connected to the num_neighbors nearest neighbors.
        (num_neighbors needs to be even!)
        Then iterate through each edge and rip it out with probability p_rewire.
        Then iterate through all the possible edges that are not present and insert with such a probability,
        that in expectation the resulting graph has the same number of edges again.
        For p=1, this yields the binomial Erdös-Renyi graph G(N, K/N).

        The random graph may contain isolated vertices, which is not allowed.
        In that case, the Generator samples until a valid network is found,
        or until max_sample_time seconds pass, in which case a RuntimeError is raised.

        Parameters
        ----------
        num_agents : int
        num_neighbors : int
        p_rewire : float
        max_sample_time : float, optional
            In seconds.
        rng : Generator, optional
            random number generator
        """
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.p_rewire = p_rewire
        self.max_sample_time = max_sample_time

        self.p_insert = (
            p_rewire * num_neighbors / (num_agents - 1 - (1 - p_rewire) * num_neighbors)
        )
        self.gnp_fun = (
            nx.erdos_renyi_graph if self.p_insert > 0.2 else nx.fast_gnp_random_graph
        )
        self.rng = rng

    def _sample_network(self):
        network = nx.watts_strogatz_graph(
            self.num_agents, self.num_neighbors, 0, seed=self.rng
        )

        # remove edges
        edges = np.array(network.edges)
        idx_to_keep = self.rng.random(edges.shape[0]) > self.p_rewire
        edges = edges[idx_to_keep, :]

        # insert edges
        network = self.gnp_fun(self.num_agents, self.p_insert, seed=self.rng)
        network.add_edges_from(edges)
        return network

    def __call__(self) -> nx.Graph:
        start = time.time()
        while True:
            network = self._sample_network()
            if nx.number_of_isolates(network) == 0:
                return network

            if time.time() - start > self.max_sample_time:
                raise RuntimeError(
                    "Timeout. Could not generate a graph without isolated vertices."
                )

    def __repr__(self) -> str:
        return f"Binomial Watts-Strogatz graph on {self.num_agents} nodes with p_rewire={self.p_rewire}"

    def abrv(self):
        return f"binWS_k{self.num_neighbors}_p{int(self.p_rewire * 100)}_N{self.num_agents}"


class BianconiBarabasiGenerator:
    def __init__(
        self, num_agents: int, m: int, lamda: float, rng: Generator = default_rng()
    ):
        """
        Generate random graphs using the Bianconi-Barabasi model.

        Every node has a fitness eta in [0,1] that is drawn randomly from the distribution with density
        p(eta) = (lambda + 1) (1 - eta)^lambda.
        Each new node i is linked to m existing nodes.
        The probability for a link between the new node i and an existing node j is proportional to
        eta_j d_j, where d_j is the degree of node j.

        For lambda > 1, the network undergoes Bose-Einstein condensation, i.e., there is one node that
        maintains a non-vanishing fraction of the links (winner takes it all).
        For lambda < 1, the fittest nodes accumulate most links,
        but every node has a vanishing fraction of links (fit get rich).

        Parameters
        ----------
        num_agents : int
        m : int
        lamda: float
        rng : Generator, optional
            random number generator
        """
        self.num_agents = num_agents
        self.m = m
        self.lamda = lamda
        self.rng = rng

    def __call__(self) -> nx.Graph:
        g = nx.star_graph(self.m)
        degrees = np.array([d for _, d in g.degree()])
        degrees = np.concatenate(
            [degrees, np.ones(self.num_agents - len(degrees)) * self.m]
        )
        fitness_values = self.rng.beta(1, self.lamda + 1, self.num_agents)

        for size_g in range(len(g), self.num_agents):
            probabilities = degrees[:size_g] * fitness_values[:size_g]
            probabilities /= np.sum(probabilities)
            nodes_to_link = self.rng.choice(
                size_g, size=self.m, replace=False, p=probabilities, shuffle=False
            )
            g.add_edges_from(zip([size_g] * self.m, nodes_to_link))

            for other in nodes_to_link:
                degrees[other] += 1

        return g

    def __repr__(self) -> str:
        return f"Bianconi-Barabasi random graph on {self.num_agents} nodes"

    def abrv(self):
        return f"bianconi_barabasi_m{self.m}_N{self.num_agents}"


def _unisolate_vertices(network: nx.Graph) -> None:
    """
    Make isolated vertices un-isolated by adding one edge to a random node.

    Parameters
    ----------
    network : nx.Graph
    """
    for i in nx.isolates(network):
        j = i
        while j == i:
            j = np.random.randint(0, network.number_of_nodes())

        network.add_edge(i, j)
