import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

from ...utils import argmatch
from ..parameters import CNVMParameters


def sample_stochastic_approximation(
    params: CNVMParameters,
    initial_states: NDArray,
    t_max: float,
    num_samples: int,
    t_eval: ArrayLike,
) -> tuple[NDArray, NDArray]:
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
    initial_states : NDArray
        Either shape = (num_opinions,) or (num_states, num_opinions)
    t_max : float
    num_samples: int
    t_eval : ArrayLike
        Array of time points where the solution should be saved,
        or number "n" in which case the solution is stored equidistantly at "n" time points.

    Returns
    -------
    tuple[NDArray, NDArray]
        (t, c),
        t.shape=(num_timesteps),
        c.shape = (num_states, num_samples, num_timesteps, num_opinions), or c.shape = (num_samples, num_timesteps, num_opinions) if a single initial state was given.
    """
    if isinstance(t_eval, float):
        raise ValueError("t_eval has to be an array of time points or an int.")
    if isinstance(t_eval, int):
        t_eval = np.linspace(0, t_max, t_eval)
    t_eval = np.array(t_eval)
    if np.min(t_eval) < 0:
        raise ValueError("The times in t_eval have to be >= 0.")
    if np.min(np.diff(t_eval)) <= 0:
        raise ValueError("The times in t_eval have to be increasing.")

    if initial_states.ndim == 1:
        return _sample_many(
            initial_states,
            params.num_agents,
            params.r,
            params.r_tilde,
            t_eval,
            num_samples,
        )

    num_states = initial_states.shape[0]
    c = np.zeros(
        (
            num_states,
            num_samples,
            num_timesteps + 1,
            initial_states.shape[1],
        )
    )
    t = np.zeros(num_timesteps + 1)
    for i in range(num_states):
        t, c[i] = _sample_many(
            initial_states[i],
            t_max,
            params.num_agents,
            params.r,
            params.r_tilde,
            num_timesteps,
            num_samples,
        )
    return t, c


@njit(parallel=True, cache=True)
def _sample_many(
    initial_state: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
    t_eval: NDArray,
    num_samples: int,
) -> tuple[NDArray, NDArray]:
    c_out = np.zeros((num_samples, t_eval.shape[0], initial_state.shape[0]))

    for i in prange(num_samples):
        t, c = _simulate(
            initial_state,
            t_eval,
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
    initial_state: NDArray,
    t_eval: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
):
    t = 0
    c = initial_state.copy()
    num_opinions = c.shape[0]

    t_list = [0.0]
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
