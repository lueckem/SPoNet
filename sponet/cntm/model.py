import numpy as np
from numba import njit
from numba.typed import List
from numpy.random import Generator, default_rng

from ..sampling import sample_randint
from ..utils import calculate_neighbor_list, mask_subsequent_duplicates
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
        self.neighbor_list = List(calculate_neighbor_list(params.network))

        self.noise_prob = self.params.r_tilde / (self.params.r + self.params.r_tilde)
        self.next_event_rate = 1 / (
            self.params.num_agents * (self.params.r + self.params.r_tilde)
        )

    def simulate(
        self,
        t_max: float,
        x_init: np.ndarray,
        len_output: int = None,
        rng: Generator = default_rng(),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model from t=0 to t=t_max.

        Parameters
        ----------
        t_max : float
        x_init : np.ndarray
            shape=(num_agents,)
        len_output : int, optional
            number of snapshots to output, as equidistantly spaced as possible between 0 and t_max
        rng : Generator, optional
            random number generator

        Returns
        -------
        tuple[np.ndarray]
            t_traj (shape=(?,)), x_traj (shape=(?,num_agents))
        """
        x = np.copy(x_init).astype(float)

        t_delta = 0 if len_output is None else t_max / (len_output - 1)

        t_traj, x_traj = _simulate_numba(
            x,
            t_delta,
            self.next_event_rate,
            self.noise_prob,
            t_max,
            self.params.num_agents,
            self.params.threshold_01,
            self.params.threshold_10,
            self.neighbor_list,
            rng,
        )

        t_traj = np.array(t_traj)
        x_traj = np.array(x_traj, dtype=int)
        if len_output is None:
            # remove duplicate subsequent states
            mask = mask_subsequent_duplicates(x_traj)
            x_traj = x_traj[mask]
            t_traj = t_traj[mask]

        return t_traj, x_traj


@njit
def _simulate_numba(
    x,
    t_delta,
    next_event_rate,
    noise_prob,
    t_max,
    num_agents,
    threshold_01,
    threshold_10,
    neighbor_list,
    rng,
):
    x_traj = [np.copy(x)]
    t = 0
    t_traj = [0]

    t_store = t_delta
    while t < t_max:
        t += rng.exponential(next_event_rate)  # time of next event
        agent = sample_randint(num_agents, rng)  # agent of next event

        if rng.random() < noise_prob:  # noise
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
                    x[agent] = other_opinion

        if t >= t_store:
            t_store += t_delta
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj
