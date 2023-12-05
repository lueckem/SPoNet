import numpy as np
from numba import njit, prange

from ..parameters import CNVMParameters
from ...utils import argmatch


def sample_stochastic_approximation(
    params: CNVMParameters,
    initial_state: np.ndarray,
    max_time: float,
    num_timesteps: int,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the opinion shares directly.

    Assumes well-mixedness in the sense that the propensity of
    reaction m -> n is given by
    c[m] * (r[m, n] * c[n] + r_tilde[m, n]).

    This is only true for complete networks.
    For other networks the quality of this approximation may vary.

    Parameters
    ----------
    params : CNVMParameters
    initial_state : np.ndarray
        Initial shares c of shape (num_opinions,).
    max_time : float
    num_timesteps : int
        Number of states in returned trajectory, at equidistant times.
    num_samples: int

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        t with shape (num_timesteps,), c with shape (num_timesteps, num_opinions)
    """
    return _sample_many(
        initial_state,
        max_time,
        params.num_agents,
        params.r,
        params.r_tilde,
        num_timesteps,
        num_samples,
    )


@njit(parallel=True)
def _sample_many(
    initial_state: np.ndarray,
    t_max: float,
    num_agents: int,
    r: np.ndarray,
    r_tilde: np.ndarray,
    num_timesteps: int,
    num_samples: int,
):
    t_delta = t_max / (5 * num_timesteps)
    t_out = np.linspace(0, t_max, num_timesteps + 1)
    c_out = np.zeros((num_samples, t_out.shape[0], initial_state.shape[0]))

    for i in prange(num_samples):
        t, c = _simulate(
            initial_state,
            t_max,
            t_delta,
            num_agents,
            r,
            r_tilde,
        )
        t = np.array(t)
        c = make_2d(c)

        t_ind = argmatch(t_out, t)
        c_out[i] = c[t_ind, :]

    return t_out, c_out


@njit
def _simulate(
    initial_state: np.ndarray,
    t_max: float,
    t_delta: float,
    num_agents: int,
    r: np.ndarray,
    r_tilde: np.ndarray,
):
    t = 0
    c = initial_state.copy()
    num_opinions = c.shape[0]

    t_list = [0]
    c_list = [initial_state.copy()]
    props = np.zeros((num_opinions, num_opinions))

    t_store = t_delta
    while t < t_max:
        _update_propensities(props, r, r_tilde, c, num_agents)

        sum_props = np.sum(props)
        t += np.random.exponential(1 / sum_props)

        cum_sum = np.cumsum(props / sum_props)
        reaction = np.searchsorted(cum_sum, np.random.random(), side="right")
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


@njit
def make_2d(arraylist):
    n = len(arraylist)
    k = arraylist[0].shape[0]
    a2d = np.zeros((n, k))
    for i in range(n):
        a2d[i] = arraylist[i]
    return a2d
