import math

import numpy as np
from numba import njit, prange
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from sponet.cnvm.parameters import CNVMParameters
from sponet.utils import t_eval_to_ndarray


# TODO: tests
def sample_hybrid_jump_diffusion(
    params: CNVMParameters,
    initial_states: ArrayLike,
    t_max: float,
    num_samples: int,
    switch_propensity_threshold: float,
    switch_exit_probability_threshold: float,
    delta_t: float | None = None,
    t_eval: ArrayLike | None = None,
    rng: Generator | None = None,
    seed: int | None = None,
    return_channel_stats: bool = False,
) -> tuple[NDArray, ...]:
    # TODO: docs
    # A channel i -> j is a jump channel if its propensity is below
    # `switch_propensity_threshold` or if the probability that a full
    # Euler-Maruyama step (over all channels) leaves the simplex in the opinion i or j,
    # computed from the current state, exceeds `switch_exit_probability_threshold`.
    # If `return_channel_stats` is True, additionally returns the time each channel i -> j
    # spent in jump mode and the number of jumps it fired over the whole simulation,
    # each with shape (num_initial_states, num_samples, n_states, n_states).
    if rng is not None:
        seed = int(rng.integers(1, 2**24))
    elif seed is None:
        seed = int(default_rng().integers(1, 2**24))

    delta_t, t_eval = _sanitize_delta_t_and_t_eval(delta_t, t_eval, t_max)

    initial_states = np.array(initial_states, ndmin=1)
    is_1d = initial_states.ndim == 1
    if is_1d:
        initial_states = np.expand_dims(initial_states, 0)

    num_states = initial_states.shape[0]
    num_time_steps = t_eval.shape[0]
    c = np.zeros(
        (
            num_states,
            num_samples,
            num_time_steps,
            initial_states.shape[1],
        )
    )
    jump_times = np.zeros(
        (num_states, num_samples, initial_states.shape[1], initial_states.shape[1])
    )
    jump_counts = np.zeros(jump_times.shape, dtype=np.int64)

    switch_exit_quantile = _exit_quantile(switch_exit_probability_threshold)

    for i in range(num_states):
        t, c[i], jump_times[i], jump_counts[i] = _numba_sample_jda(
            initial_states[i],
            delta_t,
            t_eval,
            params.num_agents,
            params.r,
            params.r_tilde,
            num_samples,
            switch_propensity_threshold,
            switch_exit_quantile,
            seed,
        )

    if is_1d:
        c = c[0]
        jump_times = jump_times[0]
        jump_counts = jump_counts[0]
    if return_channel_stats:
        return t, c, jump_times, jump_counts
    return t, c  # type: ignore


def _exit_quantile(switch_exit_probability_threshold: float) -> float:
    """
    Standard normal quantile z of the exit probability threshold.

    With mean and std of the share of an opinion after a full Euler-Maruyama step,
    P(exit) = Phi(-mean / std) > threshold  <=>  mean < -z * std.
    Comparing with the quantile avoids evaluating the normal CDF in every step.
    A threshold >= 1 yields z = inf, which disables the criterion.
    """
    return float(norm.ppf(min(max(switch_exit_probability_threshold, 0.0), 1.0)))


def _sanitize_delta_t_and_t_eval(
    delta_t: float | None, t_eval: ArrayLike | None, max_time: float
) -> tuple[float, NDArray]:
    if delta_t is None and t_eval is None:
        raise ValueError("Either `delta_t` or `t_eval` has to be provided.")

    if t_eval is not None:
        t_eval = t_eval_to_ndarray(t_eval, max_time)

        if delta_t is None:
            delta_t = np.max(np.diff(t_eval))

    if delta_t is not None and t_eval is None:
        num_steps = int(np.ceil(max_time / delta_t))
        t_eval = np.linspace(0, delta_t * num_steps, num_steps + 1)
        t_eval[-1] = max_time

    assert isinstance(t_eval, np.ndarray)
    assert isinstance(delta_t, float)
    return delta_t, t_eval


@njit(parallel=True, cache=True)
def _numba_sample_jda(
    c_init: NDArray,
    delta_t: float,
    t_eval: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
    num_samples: int,
    switch_propensity_threshold: float,
    switch_exit_quantile: float,
    seed: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    n_states = c_init.shape[0]
    c_out = np.zeros((num_samples, t_eval.shape[0], n_states))
    jump_times_out = np.zeros((num_samples, n_states, n_states))
    jump_counts_out = np.zeros((num_samples, n_states, n_states), dtype=np.int64)

    for i in prange(num_samples):
        np.random.seed(seed + i)
        c_out[i], jump_times_out[i], jump_counts_out[i] = _numba_jda(
            c_init,
            delta_t,
            t_eval,
            num_agents,
            r,
            r_tilde,
            switch_propensity_threshold,
            switch_exit_quantile,
        )

    return t_eval, c_out, jump_times_out, jump_counts_out


@njit()
def _numba_jda(
    c_init: NDArray,
    delta_t: float,
    t_eval: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
    switch_propensity_threshold: float,
    switch_exit_quantile: float,
) -> tuple[NDArray, NDArray, NDArray]:
    # Execute one simulation. Returns the stored states, shape = (len(t_eval), n_states),
    # the time each channel spent in jump mode and the number of jumps each channel fired,
    # both with shape = (n_states, n_states).

    n_states = c_init.shape[0]
    c_store = np.zeros((t_eval.shape[0], n_states))
    c_store[0] = c_init

    jump_channels = np.zeros((n_states, n_states), dtype=np.bool_)
    jump_thresholds = np.zeros((n_states, n_states))
    jump_integrated_times = np.zeros((n_states, n_states))
    jump_times = np.zeros((n_states, n_states))
    jump_counts = np.zeros((n_states, n_states), dtype=np.int64)

    propensities = np.zeros((n_states, n_states))
    exit_likely = np.zeros(n_states, dtype=np.bool_)

    t = 0.0
    next_store_index = 1
    next_t_store = t_eval[next_store_index]

    c = np.copy(c_init)

    while True:
        if t + delta_t >= next_t_store:
            this_delta_t = next_t_store - t
            store = True
        else:
            this_delta_t = delta_t
            store = False

        _numba_update_propensities(propensities, c, r, r_tilde)
        _numba_update_channels(
            c,
            propensities,
            this_delta_t,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            switch_propensity_threshold,
            switch_exit_quantile,
            num_agents,
            exit_likely,
        )
        for i in range(n_states):
            for j in range(n_states):
                if jump_channels[i, j]:
                    jump_times[i, j] += this_delta_t
        _numba_compute_timestep(
            c,
            propensities,
            this_delta_t,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            num_agents,
            jump_counts,
        )

        t += this_delta_t

        if store:
            c_store[next_store_index] = c
            next_store_index += 1
            if next_store_index >= t_eval.shape[0]:
                break
            next_t_store = t_eval[next_store_index]

    return c_store, jump_times, jump_counts


@njit()
def _numba_update_channels(
    c: NDArray,
    propensities: NDArray,
    delta_t: float,
    jump_channels: NDArray,
    jump_thresholds: NDArray,
    jump_integrated_times: NDArray,
    switch_propensity_threshold: float,
    switch_exit_quantile: float,
    num_agents: int,
    exit_likely: NDArray,
):
    """
    Decide for every channel i -> j whether it is a jump or a diffusion channel.

    A channel is a jump channel if its propensity is below `switch_propensity_threshold`
    or if the probability that an Euler-Maruyama step of size `delta_t` leaves the simplex
    in the opinion i or j exceeds the threshold whose standard normal quantile is
    `switch_exit_quantile` (see `_exit_quantile`, inf disables the second criterion).
    `exit_likely` is a work array of shape (n_states,).
    """
    n_states = c.shape[0]
    check_exit = switch_exit_quantile < np.inf
    for m in range(n_states):
        exit_likely[m] = False
        if check_exit:
            mean, std = _numba_exit_mean_std(c, propensities, delta_t, num_agents, m)
            if std > 0:
                # P(exit) = Phi(-mean / std) > threshold  <=>  mean < -quantile * std
                exit_likely[m] = mean < -switch_exit_quantile * std
            else:
                # deterministic step: P(exit) is 1 if mean < 0, else 0
                exit_likely[m] = mean < 0

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            if (
                propensities[i, j] <= switch_propensity_threshold
                or exit_likely[i]
                or exit_likely[j]
            ):
                if jump_channels[i, j] == 0:
                    # Start new jump phase
                    jump_channels[i, j] = 1
                    jump_integrated_times[i, j] = 0
                    jump_thresholds[i, j] = np.random.exponential(1)
                continue
            else:
                jump_channels[i, j] = 0

    return


@njit(inline="always")
def _numba_exit_mean_std(
    c: NDArray, propensities: NDArray, delta_t: float, num_agents: int, m: int
) -> tuple[float, float]:
    """
    Mean and standard deviation of the share of opinion m after a full Euler-Maruyama step.

    Given the current state, the new share is normally distributed with
    mean c_m + (inflow_m - outflow_m) * delta_t and variance (inflow_m + outflow_m) * delta_t / N,
    where in-/outflow are the sums of the propensities into/out of m.
    The probability to leave the simplex is P(new share < 0) = Phi(-mean / std).
    """
    n_states = c.shape[0]
    inflow = 0.0
    outflow = 0.0
    for k in range(n_states):
        if k == m:
            continue
        inflow += propensities[k, m]
        outflow += propensities[m, k]
    mean = c[m] + (inflow - outflow) * delta_t
    std = math.sqrt((inflow + outflow) * delta_t / num_agents)
    return mean, std


@njit()
def _numba_compute_timestep(
    c: NDArray,
    propensities: NDArray,
    delta_t: float,
    jump_channels: NDArray,
    jump_thresholds: NDArray,
    jump_integrated_times: NDArray,
    num_agents: int,
    jump_counts: NDArray,
):
    """
    Advance the hybrid process by one step of size `delta_t` in place.

    Diffusive channels take an Euler-Maruyama step, jump channels integrate their
    propensity and fire a jump of size 1/num_agents once the integrated time exceeds
    the jump threshold. Integrated time exceeding the threshold is carried over to the
    next jump. Increments and jumps that would leave the simplex are clipped such that
    the trajectory ends up on the boundary, without altering time.

    Parameters
    ----------
    c : NDArray
        Shape = (n_states,). Current state, overwritten with the new state.
    propensities : NDArray
        Shape = (n_states, n_states). Propensities evaluated at `c`,
        overwritten with the propensities evaluated at the new state.
    delta_t : float
    jump_channels : NDArray
        Shape = (n_states, n_states).
    jump_thresholds : NDArray
        Shape = (n_states, n_states).
    jump_integrated_times : NDArray
        Shape = (n_states, n_states).
    num_agents : int
    jump_counts : NDArray
        Shape = (n_states, n_states). Incremented in place for every fired jump.
    """
    n_states = c.shape[0]
    std = np.sqrt(delta_t)

    # Diffusion step
    for i in range(n_states):
        for j in range(n_states):
            if i == j or jump_channels[i, j]:
                continue
            prop = propensities[i, j]
            increment = prop * delta_t + np.sqrt(prop / num_agents) * np.random.normal(
                0, std
            )
            # Clip increment such that the trajectory ends up on the boundary
            increment = min(max(increment, -c[j]), c[i])
            c[i] -= increment
            c[j] += increment

    # Jump step
    for i in range(n_states):
        for j in range(n_states):
            if i == j or not jump_channels[i, j]:
                continue
            jump_integrated_times[i, j] += num_agents * propensities[i, j] * delta_t
            # The integrated time exceeding the threshold is carried over,
            # which allows multiple jumps per step.
            while jump_integrated_times[i, j] > jump_thresholds[i, j]:
                # Clip jump such that the trajectory ends up on the boundary
                jump_size = min(1 / num_agents, c[i])
                c[i] -= jump_size
                c[j] += jump_size

                jump_integrated_times[i, j] -= jump_thresholds[i, j]
                jump_thresholds[i, j] = np.random.exponential(1)
                jump_counts[i, j] += 1


@njit(inline="always")
def _numba_update_propensities(
    propensities: NDArray, c: NDArray, r: NDArray, r_tilde: NDArray
):
    n_states = c.shape[0]
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            propensities[i, j] = c[i] * (r[i, j] * c[j] + r_tilde[i, j])
