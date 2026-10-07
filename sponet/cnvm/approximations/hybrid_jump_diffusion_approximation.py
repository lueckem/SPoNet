import numpy as np
from numba import njit, prange
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray

from sponet.cnvm.parameters import CNVMParameters
from sponet.utils import t_eval_to_ndarray


# TODO: tests
def sample_hybrid_jump_diffusion(
    params: CNVMParameters,
    initial_states: ArrayLike,
    t_max: float,
    num_samples: int,
    switch_propensity_threshold: float,
    switch_boundary_thresholds: NDArray,
    delta_t: float | None = None,
    t_eval: ArrayLike | None = None,
    rng: Generator | None = None,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    # TODO: docs
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

    for i in range(num_states):
        t, c[i] = _numba_sample_jda(
            initial_states[i],
            delta_t,
            t_eval,
            params.num_agents,
            params.r,
            params.r_tilde,
            num_samples,
            switch_propensity_threshold,
            switch_boundary_thresholds,
            seed,
        )

    if is_1d:
        c = c[0]
    return t, c  # type: ignore


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
    switch_boundary_thresholds: NDArray,
    seed: int,
) -> tuple[NDArray, NDArray]:
    n_states = c_init.shape[0]
    c_out = np.zeros((num_samples, t_eval.shape[0], n_states))

    for i in prange(num_samples):
        np.random.seed(seed + i)
        c_out[i] = _numba_jda(
            c_init,
            delta_t,
            t_eval,
            num_agents,
            r,
            r_tilde,
            switch_propensity_threshold,
            switch_boundary_thresholds,
        )

    return t_eval, c_out


@njit()
def _numba_jda(
    c_init: NDArray,
    delta_t: float,
    t_eval: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
    switch_propensity_threshold: float,
    switch_boundary_thresholds: NDArray,
) -> NDArray:
    # Execute one simulation

    n_states = c_init.shape[0]
    c_store = np.zeros((t_eval.shape[0], n_states))
    c_store[0] = c_init

    jump_channels = np.zeros((n_states, n_states), dtype=np.bool_)
    jump_thresholds = np.zeros((n_states, n_states))
    jump_integrated_times = np.zeros((n_states, n_states))

    propensities = np.zeros((n_states, n_states))

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
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            switch_propensity_threshold,
            switch_boundary_thresholds,
        )
        _numba_compute_timestep(
            c,
            propensities,
            this_delta_t,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            num_agents,
        )

        t += this_delta_t

        if store:
            c_store[next_store_index] = c
            next_store_index += 1
            if next_store_index >= t_eval.shape[0]:
                break
            next_t_store = t_eval[next_store_index]

    return c_store


@njit()
def _numba_update_channels(
    c: NDArray,
    propensities: NDArray,
    jump_channels: NDArray,
    jump_thresholds: NDArray,
    jump_integrated_times: NDArray,
    switch_propensity_threshold: float,
    switch_boundary_thresholds: NDArray,
):
    n_states = c.shape[0]
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            if (
                propensities[i, j] <= switch_propensity_threshold
                or c[i] <= switch_boundary_thresholds[i, j]
                or c[j] <= switch_boundary_thresholds[j, i]
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


@njit()
def _numba_compute_timestep(
    c: NDArray,
    propensities: NDArray,
    delta_t: float,
    jump_channels: NDArray,
    jump_thresholds: NDArray,
    jump_integrated_times: NDArray,
    num_agents: int,
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
