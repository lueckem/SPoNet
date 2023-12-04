import numpy as np
from numpy.random import Generator
from numba import njit

from ..parameters import CNVMParameters
from ...sampling import sample_weighted_bisect


def sample_stochastic_approximation(
    params: CNVMParameters,
    initial_state: np.ndarray,
    max_time: float,
    len_output: int = None,
    rng: Generator = np.random.default_rng(),
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    params : CNVMParameters
    initial_state : np.ndarray
        Initial shares c.
    max_time : float
    len_output : int, optional
        Number of states to return, as equidistantly placed as possible.
        If None, the whole trajectory is returned.
    rng : Generator, optional
        random number generator

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
    """
    t_delta = 0 if len_output is None else max_time / (len_output - 1)

    t_traj, c_traj = _simulate(
        initial_state,
        max_time,
        t_delta,
        params.num_agents,
        params.r,
        params.r_tilde,
        rng,
    )
    return np.array(t_traj), np.array(c_traj)


@njit
def _simulate(
    initial_state: np.ndarray,
    t_max: float,
    t_delta: float,
    num_agents: int,
    r: np.ndarray,
    r_tilde: np.ndarray,
    rng: Generator,
):
    t = 0
    c = initial_state.copy()
    num_opinions = c.shape[0]

    t_list = []
    c_list = []
    props = np.zeros((num_opinions, num_opinions))

    t_store = t_delta
    while t < t_max:
        _update_propensities(props, r, r_tilde, c, num_agents)

        sum_props = np.sum(props)
        t += rng.exponential(1 / sum_props)

        reaction = sample_weighted_bisect(np.cumsum(props / sum_props), rng)
        m, n = reaction // num_opinions, reaction % num_opinions
        c[m] -= 1 / num_agents
        c[n] += 1 / num_agents

        if t >= t_store:
            t_store += t_delta
            t_list.append(t)
            c_list.append(c.copy())

    return t_list, c_list


@njit
def _update_propensities(
    props: np.ndarray,
    r: np.ndarray,
    r_tilde: np.ndarray,
    c: np.ndarray,
    num_agents: int,
):
    for m in range(props.shape[0]):
        for n in range(props.shape[1]):
            if m == n:
                continue
            props[m, n] = c[m] * (r[m, n] * c[n] + r_tilde[m, n])
    props *= num_agents
