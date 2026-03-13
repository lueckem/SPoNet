import numpy as np
from numba import njit, prange
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray

from sponet.utils import t_eval_to_ndarray

from ..parameters import CNVMParameters


def sample_stochastic_approximation(
    params: CNVMParameters,
    initial_states: ArrayLike,
    t_max: float,
    num_samples: int,
    t_eval: ArrayLike,
    rng: Generator | None = None,
    seed: int | None = None,
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
    rng : Generator, optional
        Random number generator.
    seed : int, optional
        Seed for random numbers. If both `rng` and `seed` are given, `rng` takes precedence.

    Returns
    -------
    tuple[NDArray, NDArray]
        (t, c),
        t.shape=(num_timesteps),
        c.shape = (num_states, num_samples, num_timesteps, num_opinions), or c.shape = (num_samples, num_timesteps, num_opinions) if a single initial state was given.
    """
    if rng is not None:
        seed = int(rng.integers(1, 2**24))
    elif seed is None:
        seed = int(default_rng().integers(1, 2**24))

    t_eval = t_eval_to_ndarray(t_eval, t_max)

    initial_states = np.array(initial_states, ndmin=1)
    is_1d = initial_states.ndim == 1
    if is_1d:
        initial_states = np.expand_dims(initial_states, 0)

    num_states = initial_states.shape[0]
    num_timesteps = t_eval.shape[0]
    c = np.zeros(
        (
            num_states,
            num_samples,
            num_timesteps,
            initial_states.shape[1],
        )
    )

    for i in range(num_states):
        c[i] = _sample_many(
            initial_states[i],
            params.num_agents,
            params.r,
            params.r_tilde,
            t_eval,
            num_samples,
            seed,
        )

    if is_1d:
        c = c[0]
    return t_eval, c


@njit(parallel=True, cache=True)
def _sample_many(
    initial_state: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
    t_eval: NDArray,
    num_samples: int,
    seed: int,
) -> NDArray:
    c_out = np.zeros((num_samples, t_eval.shape[0], initial_state.shape[0]))

    for i in prange(num_samples):
        np.random.seed(seed + i)
        c = _simulate(
            initial_state,
            t_eval,
            num_agents,
            r,
            r_tilde,
        )

        c_out[i] = c

    return c_out


@njit
def _simulate(
    initial_state: NDArray,
    t_eval: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
) -> NDArray:
    t_max = t_eval[-1]
    t = 0.0
    c = initial_state.copy()
    num_opinions = c.shape[0]

    c_store = np.zeros((t_eval.shape[0], c.shape[0]))
    c_store[0] = c
    props = np.zeros((num_opinions, num_opinions))

    t_store_idx = 1
    while t < t_max:
        _update_propensities(props, r, r_tilde, c, num_agents)

        sum_props = np.sum(props)
        t += np.random.exponential(1 / sum_props)

        cum_sum = np.cumsum(props / sum_props)
        reaction = np.searchsorted(cum_sum, np.random.random(), side="right")
        m, n = reaction // num_opinions, reaction % num_opinions
        c[m] -= 1 / num_agents
        c[n] += 1 / num_agents

        if t >= t_eval[t_store_idx]:
            c_store[t_store_idx] = c
            t_store_idx += 1

    if t_store_idx < t_eval.shape[0]:
        # `c_store` is not filled completely.
        # Simply duplicate the last value.
        # This could be done better, see CNVM.
        c_store[t_store_idx:] = c_store[t_store_idx - 1]
    return c_store


@njit
def _update_propensities(
    props: NDArray,
    r: NDArray,
    r_tilde: NDArray,
    c: NDArray,
    num_agents: int,
):
    for m in range(props.shape[0]):
        for n in range(props.shape[1]):
            if m == n:
                continue
            props[m, n] = c[m] * (r[m, n] * c[n] + r_tilde[m, n])
    props *= num_agents
