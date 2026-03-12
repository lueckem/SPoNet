import numpy as np
from numba import njit
from numba.typed.typedlist import List as NumbaList
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray

from ..sampling import build_alias_table, sample_from_alias, sample_randint
from ..utils import argmatch, store_snapshot_linspace, t_eval_to_ndarray
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

        self.neighbor_list = (
            NumbaList() if params.network is None else NumbaList(params.network)
        )

    def _calculate_degree_alpha(self) -> None:
        """
        Calculate and set self.degree_alpha.
        """
        if self.params.network is not None:
            degrees = np.array([len(nbrs) for nbrs in self.params.network])
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
        self.neighbor_list = NumbaList(self.params.network)
        self._calculate_degree_alpha()

    def update_rates(
        self,
        r: ArrayLike | None = None,
        r_tilde: ArrayLike | None = None,
    ) -> None:
        """
        Update one or both rate parameters.

        If only one argument is given, the other rate parameter stays the same.

        Parameters
        ----------
        r : ArrayLike, optional
        r_tilde : ArrayLike, optional
        """
        self.params.change_rates(r, r_tilde)

    def simulate(
        self,
        t_max: float,
        x_init: ArrayLike | None = None,
        t_eval: ArrayLike | None = None,
        rng: Generator = default_rng(),
    ) -> tuple[NDArray, NDArray]:
        """
        Simulate the model from t=0 to t=t_max.

        Parameters
        ----------
        t_max : float
        x_init : ArrayLike, optional
            Initial state, shape=(num_agents,). If no x_init is given, a random one is generated.
        t_eval : ArrayLike, optional
            Array of time points where the solution should be saved,
            or number "n" in which case the solution is stored equidistantly at "n" time points. If None, store every snapshot.
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
            x = rng.choice(
                np.arange(self.params.num_opinions, dtype=opinion_dtype),
                size=self.params.num_agents,
            )
        else:
            x = np.array(x_init, dtype=opinion_dtype)

        if t_eval is not None:
            t_eval = t_eval_to_ndarray(t_eval, t_max)

        # Call the correct simulation loop
        t_traj, x_traj = self._call_simulation_loop(x, t_max, t_eval, rng)

        if t_eval is not None and t_traj.shape[0] != t_eval.shape[0]:
            # there might be less samples than len(t_eval)
            # -> fill with duplicates
            t_ind = argmatch(t_eval, t_traj)
            t_traj = t_eval
            x_traj = x_traj[t_ind]

        return t_traj, x_traj

    def _call_simulation_loop(
        self, x: NDArray, t_max: float, t_eval: NDArray | None, rng: Generator
    ) -> tuple[NDArray, NDArray]:
        """
        Call the correct simulation loop for the provided settings.
        There is an optimized loop for complete networks,
        and different loops depending on whether all snapshots should be saved (`t_eval = None`)
        or snapshots should be saved at `t_eval`.
        """
        if self.params.network is None and t_eval is None:
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
        elif self.params.network is None and t_eval is not None:
            t_traj, x_traj = _simulate_complete_teval(
                x,
                t_eval,
                self.params.num_opinions,
                self.params.r_imit,
                self.params.r_noise,
                self.params.prob_imit,
                self.params.prob_noise,
                self.degree_alpha,
                rng,
            )
        elif t_eval is None:
            t_traj, x_traj = _simulate_all(
                x,
                t_max,
                self.params.num_opinions,
                self.neighbor_list,  # type: ignore
                self.params.r_imit,
                self.params.r_noise,
                self.params.prob_imit,
                self.params.prob_noise,
                self.degree_alpha,
                rng,
            )
        else:
            t_traj, x_traj = _simulate_teval(
                x,
                t_eval,
                self.params.num_opinions,
                self.neighbor_list,  # type: ignore
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

        update = False  # whether a state update occured in this step

        if rng.random() < noise_probability:  # noise
            agent = sample_randint(num_agents, rng)  # agent of next event
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
                update = True
        else:  # imitation
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
def _simulate_teval(
    x: NDArray,
    t_eval: NDArray,
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
    CNVM simulation, storing snapshots at `t_eval`.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    next_event_rate = 1 / (r_imit * np.sum(degree_alpha) + r_noise * num_agents)
    noise_probability = r_noise * num_agents * next_event_rate
    prob_table, alias_table = build_alias_table(degree_alpha)

    # initialize
    x_traj = [np.copy(x)]
    t = t_eval[0]
    t_traj = [t_eval[0]]

    # In the previous step, `previous_agent` switched from its `previous_opinion` to its current opinion.
    previous_agent = 0
    previous_opinion = x[0]
    previous_t = t

    # simulation loop
    t_store_idx = 1
    len_t_eval = len(t_eval)
    while t_store_idx < len_t_eval:
        if rng.random() < noise_probability:  # noise
            agent = sample_randint(num_agents, rng)  # agent of next event
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                previous_t = t
                previous_agent = agent
                previous_opinion = x[agent]
                x[agent] = new_opinion
        else:  # imitation
            agent = sample_from_alias(prob_table, alias_table, rng)
            neighbors = neighbor_list[agent]
            rand_neighbor = neighbors[sample_randint(len(neighbors), rng)]
            new_opinion = x[rand_neighbor]
            if rng.random() < prob_imit[x[agent], new_opinion]:
                previous_t = t
                previous_agent = agent
                previous_opinion = x[agent]
                x[agent] = new_opinion

        t += rng.exponential(next_event_rate)  # time of next event
        if t >= t_eval[t_store_idx]:  # store only after passing the next `t_store`
            store_snapshot_linspace(
                t,
                t_eval[t_store_idx],
                previous_t,
                x,
                previous_agent,
                previous_opinion,
                x_traj,
                t_traj,
            )
            t_store_idx += 1

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

        update = False  # whether a state update occured in this step

        if rng.random() < noise_probability:  # noise
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
                update = True
        else:  # imitation
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
def _simulate_complete_teval(
    x: NDArray,
    t_eval: NDArray,
    num_opinions: int,
    r_imit: float,
    r_noise: float,
    prob_imit: NDArray,
    prob_noise: NDArray,
    degree_alpha: NDArray,
    rng: Generator,
) -> tuple[list[float], list[NDArray]]:
    """
    CNVM simulation for complete networks, storing snapshots at `t_eval`.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    max_degree_alpha = degree_alpha[0]
    next_event_rate = 1 / (r_imit * max_degree_alpha + r_noise) / num_agents
    noise_probability = r_noise / (r_noise + r_imit * max_degree_alpha)

    # initialize
    x_traj = [np.copy(x)]
    t = t_eval[0]
    t_traj = [t_eval[0]]

    # In the previous step, `previous_agent` switched from its `previous_opinion` to its current opinion.
    previous_agent = 0
    previous_opinion = x[0]
    previous_t = t

    # simulation loop
    t_store_idx = 1
    len_t_eval = len(t_eval)
    while t_store_idx < len_t_eval:
        agent = sample_randint(num_agents, rng)  # agent of next event
        if rng.random() < noise_probability:  # noise
            new_opinion = sample_randint(num_opinions, rng)
            if rng.random() < prob_noise[x[agent], new_opinion]:
                previous_t = t
                previous_agent = agent
                previous_opinion = x[agent]
                x[agent] = new_opinion
        else:  # imitation
            neighbor = sample_randint(num_agents, rng)
            while neighbor == agent:
                neighbor = sample_randint(num_agents, rng)
            new_opinion = x[neighbor]
            if rng.random() < prob_imit[x[agent], new_opinion]:
                previous_t = t
                previous_agent = agent
                previous_opinion = x[agent]
                x[agent] = new_opinion

        t += rng.exponential(next_event_rate)  # time of next event
        if t >= t_eval[t_store_idx]:  # store only after passing the next `t_store`
            store_snapshot_linspace(
                t,
                t_eval[t_store_idx],
                previous_t,
                x,
                previous_agent,
                previous_opinion,
                x_traj,
                t_traj,
            )
            t_store_idx += 1

    return t_traj, x_traj
