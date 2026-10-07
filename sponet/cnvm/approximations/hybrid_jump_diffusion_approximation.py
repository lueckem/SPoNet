import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


def sample_hybrid_jump_diffusion() -> tuple[NDArray, NDArray]:
    # TODO: execute many simulations
    pass


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

    jump_channels = np.zeros((n_states, n_states), dtype=bool)
    jump_thresholds = np.zeros((n_states, n_states))
    jump_integrated_times = np.zeros((n_states, n_states))

    propensities = np.zeros((n_states, n_states))

    t = 0.0
    next_store_index = 1
    next_t_store = t_eval[next_store_index]

    c = np.copy(c_init)
    c_buf = np.zeros(n_states)

    while True:
        if t + delta_t >= next_t_store:
            this_delta_t = next_t_store - t
            store = True
        else:
            this_delta_t = delta_t
            store = False

        _numba_compute_propensities(propensities, c, r, r_tilde)
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
            c_buf,
            this_delta_t,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            r,
            r_tilde,
            num_agents,
        )

        if store:
            c_store[next_store_index] = c
            next_store_index += 1
            if next_store_index >= t_eval.shape[0]:
                break
            next_t_store = t_eval[next_store_index]

    return c_store


@njit()
def _numba_compute_propensities(
    propensities: NDArray, c: NDArray, r: NDArray, r_tilde: NDArray
):
    n_states = c.shape[0]
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            propensities[i, j] = c[i] * (r[i, j] * c[j] + r_tilde[i, j])


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


def _numba_compute_timestep(
    c: NDArray,
    c_buf: NDArray,
    delta_t: float,
    jump_channels: NDArray,
    jump_thresholds: NDArray,
    jump_integrated_times: NDArray,
    r: NDArray,
    r_tilde: NDArray,
    num_agents: int,
):
    pass
