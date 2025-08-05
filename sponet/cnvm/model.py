from __future__ import annotations

import numpy as np
from numba import njit
from numba.typed import List
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from ..sampling import build_alias_table, sample_from_alias, sample_randint
from ..utils import (
    argmatch,
    calculate_neighbor_list,
    mask_subsequent_duplicates,
    store_snapshot_linspace,
)
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

        # set self.degree_alpha, an array containing d(i)^(1 - alpha)
        self._calculate_degree_alpha()

        # set self.neighbor_list, a list with self.neighbor_list[i] = array of neighbors of node i
        self._calculate_neighbor_list()

    def _calculate_neighbor_list(self) -> None:
        """
        Calculate and set self.neighbor_list.
        """
        self.neighbor_list = List()  # type: ignore
        if self.params.network is not None:  # not needed for complete network
            self.neighbor_list = List(calculate_neighbor_list(self.params.network))  # type: ignore

    def _calculate_degree_alpha(self) -> None:
        """
        Calculate and set self.degree_alpha.
        """
        if self.params.network is not None:
            degrees = np.array([d for _, d in self.params.network.degree()])  # type: ignore
            if np.min(degrees) < 1:
                raise ValueError("Isolated vertices in the network are not allowed.")
            self.degree_alpha = degrees ** (1 - self.params.alpha)
        else:  # fully connected
            self.degree_alpha = np.ones(self.params.num_agents) * (
                self.params.num_agents - 1
            ) ** (1 - self.params.alpha)

    def update_network(self) -> None:
        """
        Update network from NetworkGenerator in params.
        """
        self.params.update_network_by_generator()
        self._calculate_degree_alpha()
        self._calculate_neighbor_list()

    def update_rates(
        self,
        r: float | NDArray | None = None,
        r_tilde: float | NDArray | None = None,
    ) -> None:
        """
        Update one or both rate parameters.

        If only one argument is given, the other rate parameter stays the same.

        Parameters
        ----------
        r : float | NDArray, optional
        r_tilde : float | NDArray, optional
        """
        self.params.change_rates(r, r_tilde)

    def simulate(
        self,
        t_max: float,
        x_init: NDArray | None = None,
        len_output: int | None = None,
        rng: Generator = default_rng(),
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate the model from t=0 to t=t_max.

        Parameters
        ----------
        t_max : float
        x_init : NDArray, optional
            Initial state, shape=(num_agents,). If no x_init is given, a random one is generated.
        len_output : int, optional
            Number of snapshots to output, as equidistantly spaced as possible between 0 and t_max.
            Needs to be at least 2 (for initial value and final value).
        rng : Generator, optional
            random number generator

        Returns
        -------
        tuple[NDArray, NDArray]
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

        # Call the correct simulation loop
        t_traj, x_traj = self._call_simulation_loop(x, t_max, len_output, rng)

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

    def _call_simulation_loop(
        self, x: NDArray, t_max: float, len_output: int | None, rng: Generator
    ) -> tuple[NDArray, NDArray]:
        """
        Call the correct simulation loop for the provided settings.
        There is an optimized loop for complete networks,
        and different loops depending on whether all snapshots should be saved (`len_output = None`)
        or `len_output` equidistantly spaced snapshots should be saved.
        """
        t_delta = 0 if len_output is None else t_max / (len_output - 1)

        if self.params.network is None and len_output is None:
            t_traj, x_traj = _simulate_complete_all(
                x,
                t_max,
                self.params.num_opinions,
                self.params.r_imit,
                self.params.r_noise,
                self.params.prob_imit,
                self.params.prob_noise,
                self.degree_alpha,
                rng,
            )
        elif self.params.network is None and len_output is not None:
            t_traj, x_traj = _simulate_complete_linspace(
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
        elif len_output is None:
            t_traj, x_traj = _simulate_all(
                x,
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
        else:
            t_traj, x_traj = _simulate_linspace(
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
        return np.array(t_traj), np.array(x_traj)


@njit(cache=True)
def _simulate_all(
    x: NDArray,
    t_max: float,
    num_opinions: int,
    neighbor_list: list,
    r_imit: float,
    r_noise: float,
    prob_imit: NDArray,
    prob_noise: NDArray,
    degree_alpha: NDArray,
    rng: Generator,
) -> tuple[list[float], list[NDArray]]:
    """
    CNVM simulation, storing all snapshots.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    next_event_rate = 1 / (r_imit * np.sum(degree_alpha) + r_noise * num_agents)
    noise_probability = r_noise * num_agents * next_event_rate
    prob_table, alias_table = build_alias_table(degree_alpha)

    # initialize
    x_traj = [np.copy(x)]
    t = 0.0
    t_traj = [0.0]

    # simulation loop
    while t < t_max:
        t += rng.exponential(next_event_rate)  # time of next event
        noise = True if rng.random() < noise_probability else False

        # TODO: Because only states with an update are stored,
        # we do not need to mask the duplicates later anymore!
        update = False  # whether a state update occured in this step

        if noise:
            agent = sample_randint(num_agents, rng)  # agent of next event
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
                update = True
        else:
            agent = sample_from_alias(prob_table, alias_table, rng)
            neighbors = neighbor_list[agent]
            rand_neighbor = neighbors[sample_randint(len(neighbors), rng)]
            new_opinion = x[rand_neighbor]
            if rng.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion
                update = True

        if update:  # store every step (but only when the state changed)
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj


@njit(cache=True)
def _simulate_linspace(
    x: NDArray,
    t_delta: float,
    t_max: float,
    num_opinions: int,
    neighbor_list: list,
    r_imit: float,
    r_noise: float,
    prob_imit: NDArray,
    prob_noise: NDArray,
    degree_alpha: NDArray,
    rng: Generator,
) -> tuple[list[float], list[NDArray]]:
    """
    CNVM simulation, storing snapshots equidistantly with `t_delta`.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    next_event_rate = 1 / (r_imit * np.sum(degree_alpha) + r_noise * num_agents)
    noise_probability = r_noise * num_agents * next_event_rate
    prob_table, alias_table = build_alias_table(degree_alpha)

    # initialize
    x_traj = [np.copy(x)]
    t = 0.0
    t_traj = [0.0]

    # In the previous step, `previous_agent` switched from its `previous_opinion` to its current opinion.
    previous_agent = 0
    previous_opinion = x[0]

    # simulation loop
    t_store = t_delta
    while t < t_max:
        previous_t = t
        t += rng.exponential(next_event_rate)  # time of next event
        noise = True if rng.random() < noise_probability else False

        if noise:
            agent = sample_randint(num_agents, rng)  # agent of next event
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                previous_agent = agent
                previous_opinion = x[agent]
                x[agent] = new_opinion
        else:
            agent = sample_from_alias(prob_table, alias_table, rng)
            neighbors = neighbor_list[agent]
            rand_neighbor = neighbors[sample_randint(len(neighbors), rng)]
            new_opinion = x[rand_neighbor]
            if rng.random() < prob_imit[x[agent], new_opinion]:
                previous_agent = agent
                previous_opinion = x[agent]
                x[agent] = new_opinion

        if t >= t_store:  # store only after passing the next `t_store`
            store_snapshot_linspace(
                t,
                t_store,
                previous_t,
                x,
                previous_agent,
                previous_opinion,
                x_traj,
                t_traj,
            )
            t_store += t_delta

    return t_traj, x_traj


@njit(cache=True)
def _simulate_complete_all(
    x: NDArray,
    t_max: float,
    num_opinions: int,
    r_imit: float,
    r_noise: float,
    prob_imit: NDArray,
    prob_noise: NDArray,
    degree_alpha: NDArray,
    rng: Generator,
) -> tuple[list[float], list[NDArray]]:
    """
    CNVM simulation for complete networks, storing all snapshots.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    max_degree_alpha = degree_alpha[0]
    next_event_rate = 1 / (r_imit * max_degree_alpha + r_noise) / num_agents
    noise_probability = r_noise / (r_noise + r_imit * max_degree_alpha)

    # initialize
    x_traj = [np.copy(x)]
    t = 0.0
    t_traj = [0.0]

    # simulation loop
    while t < t_max:
        t += rng.exponential(next_event_rate)  # time of next event
        agent = sample_randint(num_agents, rng)  # agent of next event
        noise = True if rng.random() < noise_probability else False

        update = False  # whether a state update occured in this step

        if noise:
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
                update = True
        else:
            neighbor = sample_randint(num_agents, rng)
            while neighbor == agent:
                neighbor = sample_randint(num_agents, rng)
            new_opinion = x[neighbor]
            if rng.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion
                update = True

        if update:  # store every step (but only when the state changed)
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj


@njit(cache=True)
def _simulate_complete_linspace(
    x: NDArray,
    t_delta: float,
    t_max: float,
    num_opinions: int,
    r_imit: float,
    r_noise: float,
    prob_imit: NDArray,
    prob_noise: NDArray,
    degree_alpha: NDArray,
    rng: Generator,
) -> tuple[list[float], list[NDArray]]:
    """
    CNVM simulation for complete networks, storing snapshots equidistantly with `t_delta`.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    max_degree_alpha = degree_alpha[0]
    next_event_rate = 1 / (r_imit * max_degree_alpha + r_noise) / num_agents
    noise_probability = r_noise / (r_noise + r_imit * max_degree_alpha)

    # initialize
    x_traj = [np.copy(x)]
    t = 0.0
    t_traj = [0.0]

    # In the previous step, `previous_agent` switched from its `previous_opinion` to its current opinion.
    previous_agent = 0
    previous_opinion = x[0]

    # simulation loop
    t_store = t_delta
    while t < t_max:
        previous_t = t
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

        if t >= t_store:  # store only after passing the next `t_store`
            store_snapshot_linspace(
                t,
                t_store,
                previous_t,
                x,
                previous_agent,
                previous_opinion,
                x_traj,
                t_traj,
            )
            t_store += t_delta

    return t_traj, x_traj
