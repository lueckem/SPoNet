import numpy as np
from numba import njit
from numba.typed import List
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray

from ..sampling import sample_randint
from ..utils import (
    argmatch,
    calculate_neighbor_list,
    store_snapshot_linspace,
    t_eval_to_ndarray,
)
from .parameters import CNTMParameters


class CNTM:
    def __init__(self, params: CNTMParameters):
        """
        Continuous-time Noisy Threshold Model.

        Parameters
        ----------
        params : CNTMParameters
        """
        self.params = params

        # self.neighbor_list[i] = array of neighbors of node i
        self.neighbor_list = List(calculate_neighbor_list(params.network))  # type: ignore

        self.noise_prob = self.params.r_tilde / (self.params.r + self.params.r_tilde)
        self.next_event_rate = 1 / (
            self.params.num_agents * (self.params.r + self.params.r_tilde)
        )

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
        x_init : ArrayLike
            shape=(num_agents,)
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

        opinion_dtype = np.min_scalar_type(1)
        if x_init is None:
            x = rng.choice(
                np.arange(self.params.num_opinions, dtype=opinion_dtype),
                size=self.params.num_agents,
            )
        else:
            x = np.array(x_init, dtype=opinion_dtype)

        if t_eval is not None:
            t_eval = t_eval_to_ndarray(t_eval, t_max)

        if t_eval is None:
            t_traj, x_traj = _simulate_all(
                x,
                self.next_event_rate,
                self.noise_prob,
                t_max,
                self.params.threshold_01,
                self.params.threshold_10,
                self.neighbor_list,
                rng,
            )
        else:
            t_traj, x_traj = _simulate_teval(
                x,
                t_eval,
                self.next_event_rate,
                self.noise_prob,
                self.params.threshold_01,
                self.params.threshold_10,
                self.neighbor_list,
                rng,
            )

        t_traj = np.array(t_traj)
        x_traj = np.array(x_traj)

        if t_eval is not None and t_traj.shape[0] != t_eval.shape[0]:
            # there might be less samples than len(t_eval)
            # -> fill with duplicates
            t_ind = argmatch(t_eval, t_traj)
            t_traj = t_eval
            x_traj = x_traj[t_ind]

        return t_traj, x_traj


@njit(cache=True)
def _simulate_all(
    x: NDArray,
    next_event_rate: float,
    noise_prob: float,
    t_max: float,
    threshold_01: float,
    threshold_10: float,
    neighbor_list: list,
    rng: Generator,
) -> tuple[list[float], list[NDArray]]:
    """
    CNTM simulation, storing all snapshots.
    """
    num_agents = x.shape[0]
    x_traj = [np.copy(x)]
    t = 0.0
    t_traj = [0.0]

    while t < t_max:
        t += rng.exponential(next_event_rate)  # time of next event
        agent = sample_randint(num_agents, rng)  # agent of next event

        update = False  # whether a state update occured in this step

        if rng.random() < 0.5 * noise_prob:  # noise
            # the 0.5 is because there is a 50% change to randomly get the same opinion,
            # in which case nothing would happen
            x[agent] = 1 - x[agent]
            update = True
        else:
            neighbors = neighbor_list[agent]
            if len(neighbors) > 0:
                other_opinion = 1 - x[agent]
                share_other_opinion = 0
                for j in neighbors:
                    if x[j] == other_opinion:
                        share_other_opinion += 1
                share_other_opinion /= len(neighbors)

                threshold = threshold_10 if x[agent] == 1 else threshold_01
                if share_other_opinion >= threshold:
                    x[agent] = other_opinion
                    update = True

        if update:
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj


@njit(cache=True)
def _simulate_teval(
    x: NDArray,
    t_eval: NDArray,
    next_event_rate: float,
    noise_prob: float,
    threshold_01: float,
    threshold_10: float,
    neighbor_list: list,
    rng: Generator,
) -> tuple[list[float], list[NDArray]]:
    """
    CNTM simulation, storing snapshots at `t_eval`.
    """
    num_agents = x.shape[0]
    x_traj = [np.copy(x)]
    t = t_eval[0]
    t_traj = [t_eval[0]]

    # In the previous step, `previous_agent` switched from its `previous_opinion` to its current opinion.
    previous_agent = 0
    previous_opinion = x[0]
    previous_t = t

    t_store_idx = 1
    len_t_eval = len(t_eval)
    while t_store_idx < len_t_eval:
        agent = sample_randint(num_agents, rng)  # agent of next event
        if rng.random() < noise_prob:  # noise
            previous_t = t
            previous_agent = agent
            previous_opinion = x[agent]
            x[agent] = sample_randint(2, rng)
        else:
            neighbors = neighbor_list[agent]
            if len(neighbors) > 0:
                other_opinion = 1 - x[agent]
                share_other_opinion = 0
                for j in neighbors:
                    if x[j] == other_opinion:
                        share_other_opinion += 1
                share_other_opinion /= len(neighbors)

                threshold = threshold_10 if x[agent] == 1 else threshold_01
                if share_other_opinion >= threshold:
                    previous_t = t
                    previous_agent = agent
                    previous_opinion = x[agent]
                    x[agent] = other_opinion

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
