from __future__ import annotations
import numpy as np
from numba import njit
from numba.typed import List

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
            for i in range(self.params.num_agents):
                self.neighbor_list.append(
                    np.array(list(self.params.network.neighbors(i)), dtype=int)
                )

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
        self, t_max: float, x_init: np.ndarray = None, len_output: int = None
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

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            t_traj (shape=(?,)), x_traj (shape=(?,num_agents))
        """
        if self.params.network_generator is not None:
            self.update_network()

        if x_init is None:
            x_init = np.random.choice(
                np.arange(self.params.num_opinions), size=self.params.num_agents
            )
        x = np.copy(x_init).astype(int)

        t_delta = 0 if len_output is None else t_max / (len_output - 1)

        # for complete networks, we have a separate implementation
        if self.params.network is None:
            t_traj, x_traj = _numba_simulate_const_complete(
                x,
                t_delta,
                t_max,
                self.params.num_opinions,
                self.params.r_imit,
                self.params.r_noise,
                self.params.prob_imit,
                self.params.prob_noise,
                self.degree_alpha,
            )

        else:
            # decide which version is likely faster, log-update or constant-update
            val = np.mean(self.degree_alpha / np.max(self.degree_alpha))
            if 1 / val < np.log2(self.params.num_agents):
                sim_function = _numba_simulate_const
            else:
                sim_function = _numba_simulate_log

            t_traj, x_traj = sim_function(
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
            )

        return np.array(t_traj), np.array(x_traj, dtype=int)

    def simulate_log_complexity(
        self, t_max: float, x_init: np.ndarray = None, len_output: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model from t=0 to t=t_max using the log-complexity update.

        Depending on the model parameters, the log-complexity update can be faster
        than the constant-complexity update because the constant-complexity update
        more often performs iterations that do not result in a transition.


        Parameters
        ----------
        t_max : float
        x_init : np.ndarray, optional
            Initial state, shape=(num_agents,). If no x_init is given, a random one is generated.
        len_output : int, optional
            Number of snapshots to output, as equidistantly spaced as possible between 0 and t_max.
            Needs to be at least 2 (for initial value and final value).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            t_traj (shape=(?,)), x_traj (shape=(?,num_agents))
        """
        if self.params.network_generator is not None:
            self.update_network()

        if x_init is None:
            x_init = np.random.choice(
                np.arange(self.params.num_opinions), size=self.params.num_agents
            )
        x = np.copy(x_init).astype(int)

        t_delta = 0 if len_output is None else t_max / (len_output - 1)

        t_traj, x_traj = _numba_simulate_log(
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
        )

        return np.array(t_traj), np.array(x_traj, dtype=int)

    def simulate_const_complexity(
        self, t_max: float, x_init: np.ndarray = None, len_output: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model from t=0 to t=t_max using the constant-complexity update.

        Depending on the model parameters, the log-complexity update can be faster
        than the constant-complexity update because the constant-complexity update
        more often performs iterations that do not result in a transition.


        Parameters
        ----------
        t_max : float
        x_init : np.ndarray, optional
            Initial state, shape=(num_agents,). If no x_init is given, a random one is generated.
        len_output : int, optional
            Number of snapshots to output, as equidistantly spaced as possible between 0 and t_max.
            Needs to be at least 2 (for initial value and final value).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            t_traj (shape=(?,)), x_traj (shape=(?,num_agents))
        """
        if self.params.network_generator is not None:
            self.update_network()

        if x_init is None:
            x_init = np.random.choice(
                np.arange(self.params.num_opinions), size=self.params.num_agents
            )
        x = np.copy(x_init).astype(int)

        t_delta = 0 if len_output is None else t_max / (len_output - 1)

        t_traj, x_traj = _numba_simulate_const(
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
        )

        return np.array(t_traj), np.array(x_traj, dtype=int)


@njit()
def _rand_index_numba(prob_cum_sum) -> int:
    """
    Sample random index 0 <= i < len(prob_cumsum) according to probability distribution.

    Parameters
    ----------
    prob_cum_sum : np.ndarray
        1D array containing the cumulative probabilities, i.e.,
        the first entry is the probability of choosing index 0,
        the second entry the probability of choosing index 0 or 1, and so on.
        The last entry is 1.

    Returns
    -------
    int
    """
    return np.searchsorted(prob_cum_sum, np.random.random(), side="right")


@njit()
def _numba_simulate_log(
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
):
    """
    CNVM simulation with log-complexity update.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    next_event_rate = 1 / (r_imit * np.sum(degree_alpha) + r_noise * num_agents)
    noise_probability = r_noise * num_agents * next_event_rate
    prob_cum_sum = np.cumsum(degree_alpha / np.sum(degree_alpha))

    # initialize
    x_traj = [np.copy(x)]
    t = 0
    t_traj = [0]

    # simulation loop
    t_store = t_delta
    while t < t_max:
        t += np.random.exponential(next_event_rate)  # time of next event
        noise = True if np.random.random() < noise_probability else False

        if noise:
            agent = np.random.randint(0, num_agents)  # agent of next event
            new_opinion = np.random.randint(0, num_opinions)
            if np.random.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
        else:
            agent = _rand_index_numba(prob_cum_sum)
            neighbors = neighbor_list[agent]
            new_opinion = x[np.random.choice(neighbors)]
            if np.random.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion

        if t >= t_store:
            t_store += t_delta
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj


@njit()
def _numba_simulate_const(
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
):
    """
    CNVM simulation with constant-complexity update.
    """
    # pre-calculate some values
    num_agents = x.shape[0]
    max_degree_alpha = np.max(degree_alpha)
    next_event_rate = 1 / (r_imit * max_degree_alpha + r_noise) / num_agents
    noise_probability = r_noise / (r_noise + r_imit * max_degree_alpha)
    prob_factor = degree_alpha / max_degree_alpha

    # initialize
    x_traj = [np.copy(x)]
    t = 0
    t_traj = [0]

    # simulation loop
    t_store = t_delta
    while t < t_max:
        t += np.random.exponential(next_event_rate)  # time of next event
        agent = np.random.randint(0, num_agents)  # agent of next event
        noise = True if np.random.random() < noise_probability else False

        if noise:
            new_opinion = np.random.randint(0, num_opinions)
            if np.random.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
        elif np.random.random() < prob_factor[agent]:
            neighbors = neighbor_list[agent]
            new_opinion = x[np.random.choice(neighbors)]
            if np.random.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion

        if t >= t_store:
            t_store += t_delta
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj


@njit()
def _numba_simulate_const_complete(
    x: np.ndarray,
    t_delta: float,
    t_max: float,
    num_opinions: int,
    r_imit: float,
    r_noise: float,
    prob_imit: np.ndarray,
    prob_noise: np.ndarray,
    degree_alpha: np.ndarray,
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
        t += np.random.exponential(next_event_rate)  # time of next event
        agent = np.random.randint(0, num_agents)  # agent of next event
        noise = True if np.random.random() < noise_probability else False

        if noise:
            new_opinion = np.random.randint(0, num_opinions)
            if np.random.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
        else:
            neighbor = np.random.randint(0, num_agents)
            while neighbor == agent:
                neighbor = np.random.randint(0, num_agents)
            new_opinion = x[neighbor]
            if np.random.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion

        if t >= t_store:
            t_store += t_delta
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj
