from collections.abc import Callable
from functools import wraps
from typing import Protocol, TypeVar, cast

import networkx as nx
import numpy as np
from numba import njit
from numba.typed import List
from numpy.typing import ArrayLike, NDArray

from sponet.utils import calculate_neighbor_list

from .cnvm.parameters import CNVMParameters


class CollectiveVariable(Protocol):
    dimension: int

    def __call__(self, x: NDArray) -> NDArray:
        """
        Parameters
        ----------
        x : NDArray
            Single state with shape=(num_agents,)
            or multiple states with shape=(num_states, num_agents).

        Returns
        -------
        NDArray
            States projected down via the collective variable.
            For a single state output has shape = (self.dimension,).
            For multiple states output has shape = (num_states, self.dimension).
        """
        ...


Method = TypeVar("Method", bound=Callable[..., NDArray])


# The `Method` ensures to the type checker that the exact signature
# is preserved.
def handle_1d(func: Method) -> Method:
    """
    This decorator can be used on a method that expects a 2D input
    of shape (samples, d1) and returns a 2D output of shape (samples, d2).
    As a result, the method will also work on 1D input of shape (d1,) and
    return 1D output of shape (d2,).
    It simply wraps the 1D array in an additional dimension, calls the method,
    and removes the additional dimension.
    """

    @wraps(func)
    def wrapped_f(self, x: NDArray) -> NDArray:
        is_1d = x.ndim == 1
        if is_1d:
            x = np.expand_dims(x, 0)
        c = func(self, x)
        if is_1d:
            c = c[0, :]
        return c

    return cast(Method, wrapped_f)


class OpinionShares:
    def __init__(
        self,
        num_opinions: int,
        normalize: bool = False,
        weights: ArrayLike | None = None,
        idx_to_return: ArrayLike | None = None,
    ):
        """
        Calculate the opinion counts/ percentages, i.e., how often each opinion is present in x.

        Parameters
        ----------
        num_opinions : int
        normalize : bool, optional
            If true return percentages, else counts.
        weights : NDArray, optional
            Weight for each agent's opinion, shape=(num_agents,). Default: Each agent has weight 1.
            Negative weights are allowed.
        idx_to_return : ArrayLike, optional
            Shares of which opinions to return. Default: all opinions.
            Example: idx_to_return=0 means that only the count of opinion 0 is returned.
        """
        self.num_opinions = num_opinions
        self.weights = np.array(weights) if weights is not None else None
        self.normalize = normalize
        self.normalization = np.sum(np.abs(weights)) if weights is not None else None

        if idx_to_return is None:
            self.idx_to_return = np.arange(num_opinions)
        else:
            self.idx_to_return = np.atleast_1d(np.array(idx_to_return))

        self.dimension = len(self.idx_to_return)

    @handle_1d
    def __call__(self, x: NDArray) -> NDArray:
        """
        Parameters
        ----------
        x : NDArray
            Single state with shape=(num_agents,)
            or multiple states with shape=(num_states, num_agents).

        Returns
        -------
        NDArray
            States projected down via the collective variable.
            For a single state output has shape = (self.dimension,).
            For multiple states output has shape = (num_states, self.dimension).
        """
        # x has shape (num_states, num_agents), see @handle_1d
        num_agents = x.shape[1]
        x_agg = _opinion_shares_numba(x.astype(int), self.num_opinions, self.weights)
        x_agg = x_agg[:, self.idx_to_return]

        if self.normalize:
            if self.normalization is None:
                x_agg /= num_agents
            else:
                x_agg /= self.normalization
        return x_agg


class DegreeWeightedOpinionShares(OpinionShares):
    def __init__(
        self,
        num_opinions: int,
        network: nx.Graph,
        normalize: bool = False,
        idx_to_return: ArrayLike | None = None,
    ):
        """
        Calculate the degree-weighted opinion counts/ percentages.

        Parameters
        ----------
        num_opinions : int
        network: nx.Graph
        normalize : bool, optional
            If true return percentages, else counts.
        idx_to_return : ArrayLike, optional
            Shares of which opinions to return. Default: all opinions.
            Example: idx_to_return=0 means that only the count of opinion 0 is returned.
        """
        weights = np.array([d for _, d in network.degree()])  # type: ignore
        super().__init__(num_opinions, normalize, weights, idx_to_return)


class OpinionSharesByDegree:
    def __init__(
        self,
        num_opinions: int,
        network: nx.Graph,
        normalize: bool = False,
        idx_to_return: ArrayLike | None = None,
    ):
        """
        Calculate the count of each opinion by degree.

        The output has dimension idx_to_return * number of different degrees.
        For example, the first idx_to_return entries will represent the counts for nodes with the smallest degree.

        Parameters
        ----------
        num_opinions : int
        network : nx.Graph
        normalize : bool, optional
            If true return percentages, else counts.
            The normalization is done within each group of nodes with the same degree.
        idx_to_return : ArrayLike, optional
            Shares of which opinions to return. Default: all opinions.
        """
        self.degrees_of_nodes = np.array([d for _, d in network.degree()])  # type: ignore
        self.degrees = np.sort(np.unique(self.degrees_of_nodes))
        self.num_opinions = num_opinions
        self.normalize = normalize

        if idx_to_return is None:
            self.idx_to_return = np.arange(num_opinions)
        else:
            self.idx_to_return = np.atleast_1d(np.array(idx_to_return))

        self.dimension = len(self.idx_to_return) * self.degrees.shape[0]

    @handle_1d
    def __call__(self, x: NDArray) -> NDArray:
        """
        Parameters
        ----------
        x : NDArray
            Single state with shape=(num_agents,)
            or multiple states with shape=(num_states, num_agents).

        Returns
        -------
        NDArray
            States projected down via the collective variable.
            For a single state output has shape = (self.dimension,).
            For multiple states output has shape = (num_states, self.dimension).
        """
        # x has shape (num_states, num_agents), see @handle_1d
        cv = np.zeros((x.shape[0], self.dimension))
        num_agents = x.shape[1]
        x_int = x.astype(int)

        weights = np.zeros(num_agents)
        for i, deg in enumerate(self.degrees):
            weights[:] = 0
            weights[np.nonzero(self.degrees_of_nodes == deg)] = 1
            x_agg = _opinion_shares_numba(x_int, self.num_opinions, weights)
            x_agg = x_agg[:, self.idx_to_return]
            if self.normalize:
                x_agg /= np.sum(weights)
            cv[:, i * len(self.idx_to_return) : (i + 1) * len(self.idx_to_return)] = (
                x_agg
            )
        return cv


class CompositeCollectiveVariable:
    def __init__(self, collective_variables: list[CollectiveVariable]):
        """
        Concatenate multiple collective variables.

        Typical use-case: CV1 measures the share of opinion 1 in one part of the network,
        CV2 in a different part of the network (both built via OpinionShares class with weights).
        CompositeCollectiveVariable([CV1, CV2]) concatenates the output of the two.

        Parameters
        ----------
        collective_variables : list
        """
        self.collective_variables = collective_variables
        self.dimension = sum([cv.dimension for cv in collective_variables])

    @handle_1d
    def __call__(self, x: NDArray) -> NDArray:
        """
        Parameters
        ----------
        x : NDArray
            Single state with shape=(num_agents,)
            or multiple states with shape=(num_states, num_agents).

        Returns
        -------
        NDArray
            States projected down via the collective variable.
            For a single state output has shape = (self.dimension,).
            For multiple states output has shape = (num_states, self.dimension).
        """
        # x has shape (num_states, num_agents), see @handle_1d
        return np.concatenate([cv(x) for cv in self.collective_variables], axis=1)


class Interfaces:
    def __init__(self, network: nx.Graph, normalize: bool = False):
        """
        Count the number of interfaces between opinion 0 and 1.

        Can not be used when there are more than these two opinions.

        Parameters
        ----------
        network : nx.Graph
        normalize : bool, optional
            Normalize by dividing by the number of edges in the network.
        """
        self.dimension = 1
        self.normalize = normalize
        self.network = network

    def __call__(self, x_traj: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x_traj : np.ndarray
            trajectory of CNVM, shape = (?, num_agents).

        Returns
        -------
        np.ndarray
            trajectory projected down via the collective variable, shape = (?, self.dimension)
        """
        if np.max(x_traj) > 1:
            raise ValueError("Interfaces can only be used for 2 opinions.")
        out = np.zeros((x_traj.shape[0], 1))

        for i in range(x_traj.shape[0]):
            for u, v in self.network.edges:
                if x_traj[i, u] != x_traj[i, v]:
                    out[i, 0] += 1

        if self.normalize:
            out /= self.network.number_of_edges()

        return out


class Propensities:
    def __init__(self, params: CNVMParameters, normalize: bool = False):
        """
        The propensities are defined as cumulative transition rates in the system.

        Only implemented for 2 opinions, 0 and 1.
        Output 2-dimensional, (prop_01, prop_10).

        The propensity prop_mn is defined as
        sum_i^N ( r[m, n] * d(i,n) / (d(i)^alpha) + r_tilde[m, n] ).

        Parameters
        ----------
        params : CNVMParameters
        normalize : bool, optional
        """
        self.dimension = 2
        self.params = params
        self.normalize = normalize

    def __call__(self, x_traj: np.ndarray) -> np.ndarray:
        if self.params.network is None:
            degree_alpha = (self.params.num_agents - 1) ** self.params.alpha
            out = _propensities_complete_numba(
                x_traj, degree_alpha, self.params.r, self.params.r_tilde
            )
        else:
            degrees_alpha = (
                np.array([d for _, d in self.params.network.degree()])
                ** self.params.alpha
            )
            neighbors_list = List(calculate_neighbor_list(self.params.network))
            out = _propensities_numba(
                x_traj,
                neighbors_list,
                degrees_alpha,
                self.params.r,
                self.params.r_tilde,
            )
        if self.normalize:
            out /= self.params.num_agents
        return out


@njit(cache=True)
def _propensities_numba(x_traj, neighbor_list, degrees_alpha, r, r_tilde):
    out = np.zeros((x_traj.shape[0], 2))
    for j in range(x_traj.shape[1]):
        for i in range(x_traj.shape[0]):
            if x_traj[i, j] == 0:
                m, n = 0, 1
            else:
                m, n = 1, 0
            count_opinion_n = np.sum(x_traj[i, neighbor_list[j]] == n)
            prop_mn = r[m, n] * count_opinion_n / degrees_alpha[j] + r_tilde[m, n]
            out[i, m] += prop_mn
    return out


@njit(cache=True)
def _propensities_complete_numba(x_traj, degree_alpha, r, r_tilde):
    out = np.zeros((x_traj.shape[0], 2))
    for j in range(x_traj.shape[1]):
        for i in range(x_traj.shape[0]):
            if x_traj[i, j] == 0:
                m, n = 0, 1
            else:
                m, n = 1, 0
            count_opinion_n = np.sum(x_traj[i, :] == n)
            prop_mn = r[m, n] * count_opinion_n / degree_alpha + r_tilde[m, n]
            out[i, m] += prop_mn
    return out


@njit(cache=True)
def _opinion_shares_numba(x_traj, num_opinions, weights):
    x_agg = np.zeros((x_traj.shape[0], num_opinions))
    for i in range(x_traj.shape[0]):
        x_agg[i, :] = np.bincount(x_traj[i, :], weights=weights, minlength=num_opinions)
    return x_agg
