from __future__ import annotations

import numpy as np
from numba import njit
from numba.typed import List
from numpy.random import Generator, default_rng

from ..sampling import build_alias_table, sample_from_alias, sample_randint
from ..utils import argmatch, calculate_neighbor_list, mask_subsequent_duplicates
from .parameters import CNVMParameters


class CNVM:
    def __init__(self, params: CNVMParameters):
        """
        Continuous-time Noisy Voter Model.

        Parameters
        ----------
        params : CNVMParameters
        """
        self.params = params
        # self.neighbor_list[i] = array of neighbors of node i
        self.neighbor_list = None
        self.degree_alpha = None  # array containing d(i)^(1 - alpha)

        self._calculate_degree_alpha()
        self._calculate_neighbor_list()

    def _calculate_neighbor_list(self):
        """
        Calculate and set self.neighbor_list.
        """
        self.neighbor_list = List()
        if self.params.network is not None:  # not needed for complete network
            self.neighbor_list = List(calculate_neighbor_list(self.params.network))

    def _calculate_degree_alpha(self):
        """
        Calculate and set self.degree_alpha.
        """
        if self.params.network is not None:
            degrees = np.array([d for _, d in self.params.network.degree()])
            if np.min(degrees) < 1:
                raise ValueError("Isolated vertices in the network are not allowed.")
            self.degree_alpha = degrees ** (1 - self.params.alpha)
        else:  # fully connected
            self.degree_alpha = np.ones(self.params.num_agents) * (
                self.params.num_agents - 1
            ) ** (1 - self.params.alpha)

    def update_network(self):
        """
        Update network from NetworkGenerator in params.
        """
        self.params.update_network_by_generator()
        self._calculate_degree_alpha()
        self._calculate_neighbor_list()

    def update_rates(
        self, r: float | np.ndarray = None, r_tilde: float | np.ndarray = None
    ):
        """
        Update one or both rate parameters.

        If only one argument is given, the other rate parameter stays the same.

        Parameters
        ----------
        r : float | np.ndarray, optional
        r_tilde : float | np.ndarray
        """
        self.params.change_rates(r, r_tilde)

    def simulate(
        self,
        t_max: float,
        x_init: np.ndarray = None,
        len_output: int = None,
        rng: Generator = default_rng(),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model from t=0 to t=t_max.

        Parameters
        ----------
        t_max : float
        x_init : np.ndarray, optional
            Initial state, shape=(num_agents,). If no x_init is given, a random one is generated.
        len_output : int, optional
            Number of snapshots to output, as equidistantly spaced as possible between 0 and t_max.
            Needs to be at least 2 (for initial value and final value).
        rng : Generator, optional
            random number generator

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            t_traj (shape=(?,)), x_traj (shape=(?,num_agents))
        """
        if self.params.network_generator is not None:
            self.update_network()

        opinion_dtype = np.min_scalar_type(self.params.num_opinions - 1)

        if x_init is None:
            x_init = rng.choice(
                np.arange(self.params.num_opinions), size=self.params.num_agents
            )
        x = x_init.astype(opinion_dtype)

        t_delta = 0 if len_output is None else t_max / (len_output - 1)

        # for complete networks, we have a separate implementation
        if self.params.network is None:
            t_traj, x_traj = _numba_simulate_complete(
                x,
                t_delta,
                t_max,
                self.params.num_opinions,
                self.params.r_imit,
                self.params.r_noise,
                self.params.prob_imit,
                self.params.prob_noise,
                self.degree_alpha,
                rng,
            )

        else:
            t_traj, x_traj = _numba_simulate(
                x,
                t_delta,
                t_max,
                self.params.num_opinions,
                self.neighbor_list,
                self.params.r_imit,
                self.params.r_noise,
                self.params.prob_imit,
                self.params.prob_noise,
                self.degree_alpha,
                rng,
            )

        t_traj = np.array(t_traj)
        x_traj = np.array(x_traj, dtype=opinion_dtype)
        if len_output is None:
            # remove duplicate subsequent states
            mask = mask_subsequent_duplicates(x_traj)
            x_traj = x_traj[mask]
            t_traj = t_traj[mask]
        elif t_traj.shape[0] != len_output:
            # there might be less samples than len_output
            # -> fill them with duplicates
            t_ref = np.linspace(0, t_max, len_output)
            t_ind = argmatch(t_ref, t_traj)
            t_traj = t_ref
            x_traj = x_traj[t_ind]

        return t_traj, x_traj


@njit()
def _numba_simulate(
    x: np.ndarray,
    t_delta: float,
    t_max: float,
    num_opinions: int,
    neighbor_list: list,
    r_imit: float,
    r_noise: float,
    prob_imit: np.ndarray,
    prob_noise: np.ndarray,
    degree_alpha: np.ndarray,
    rng: Generator,
):
    """
    CNVM simulation.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    next_event_rate = 1 / (r_imit * np.sum(degree_alpha) + r_noise * num_agents)
    noise_probability = r_noise * num_agents * next_event_rate
    prob_table, alias_table = build_alias_table(degree_alpha)

    # initialize
    x_traj = [np.copy(x)]
    t = 0
    t_traj = [0]

    # simulation loop
    t_store = t_delta
    while t < t_max:
        t += rng.exponential(next_event_rate)  # time of next event
        noise = True if rng.random() < noise_probability else False

        if noise:
            agent = sample_randint(num_agents, rng)  # agent of next event
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
        else:
            agent = sample_from_alias(prob_table, alias_table, rng)
            neighbors = neighbor_list[agent]
            rand_neighbor = neighbors[sample_randint(len(neighbors), rng)]
            new_opinion = x[rand_neighbor]
            if rng.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion

        if t >= t_store:
            t_store += t_delta
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj


@njit()
def _numba_simulate_complete(
    x: np.ndarray,
    t_delta: float,
    t_max: float,
    num_opinions: int,
    r_imit: float,
    r_noise: float,
    prob_imit: np.ndarray,
    prob_noise: np.ndarray,
    degree_alpha: np.ndarray,
    rng: Generator,
):
    """
    CNVM simulation for complete networks.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    max_degree_alpha = degree_alpha[0]
    next_event_rate = 1 / (r_imit * max_degree_alpha + r_noise) / num_agents
    noise_probability = r_noise / (r_noise + r_imit * max_degree_alpha)

    # initialize
    x_traj = [np.copy(x)]
    t = 0
    t_traj = [0]

    # simulation loop
    t_store = t_delta
    while t < t_max:
        t += rng.exponential(next_event_rate)  # time of next event
        agent = sample_randint(num_agents, rng)  # agent of next event
        noise = True if rng.random() < noise_probability else False

        if noise:
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
        else:
            neighbor = sample_randint(num_agents, rng)
            while neighbor == agent:
                neighbor = sample_randint(num_agents, rng)
            new_opinion = x[neighbor]
            if rng.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion

        if t >= t_store:
            t_store += t_delta
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj
