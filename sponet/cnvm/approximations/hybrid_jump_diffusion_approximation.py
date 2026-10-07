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


@njit()
def _numba_compute_timestep(
    c: NDArray,
    c_buf: NDArray,
    propensities: NDArray,
    delta_t: float,
    jump_channels: NDArray,
    jump_thresholds: NDArray,
    jump_integrated_times: NDArray,
    r: NDArray,
    r_tilde: NDArray,
    num_agents: int,
):
    """
    Advance the hybrid process by one step of size `delta_t` in place.

    Diffusive channels take an Euler-Maruyama step, jump channels integrate their
    propensity and fire a jump of size 1/num_agents once the integrated time reaches
    the jump threshold. Leaving the simplex is handled by clipping without altering time.

    Parameters
    ----------
    c : NDArray
        Shape = (n_states,). Current state, overwritten with the new state.
    c_buf : NDArray
        Shape = (n_states,). Buffer for the intermediate state.
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
    r : NDArray
    r_tilde : NDArray
    num_agents : int
    """
    n_states = c.shape[0]
    std = np.sqrt(delta_t)

    # Diffusion step
    c_buf[:] = c
    for i in range(n_states):
        for j in range(n_states):
            if i == j or jump_channels[i, j]:
                continue
            prop = propensities[i, j]
            increment = prop * delta_t + np.sqrt(prop / num_agents) * np.random.normal(
                0, std
            )
            c_buf[i] -= increment
            c_buf[j] += increment

    # Map diffusion step back onto the simplex if it left it
    if (c_buf < 0).any():
        np.clip(c_buf, 0, 1, out=c_buf)
        c_buf /= np.sum(c_buf)

    # Jump step
    for i in range(n_states):
        for j in range(n_states):
            if i == j or not jump_channels[i, j]:
                continue
            jump_integrated_times[i, j] += num_agents * propensities[i, j] * delta_t
            if jump_integrated_times[i, j] < jump_thresholds[i, j]:
                continue

            # Clip jump such that the trajectory ends up on the boundary
            jump_size = min(1 / num_agents, c_buf[i])
            c_buf[i] -= jump_size
            c_buf[j] += jump_size

            jump_integrated_times[i, j] = 0
            jump_thresholds[i, j] = np.random.exponential(1)

    c[:] = c_buf
    _numba_update_propensities(propensities, c, r, r_tilde)


@njit(inline="always")
def _numba_update_propensities(
    propensities: NDArray, c: NDArray, r: NDArray, r_tilde: NDArray
):
    n_states = c.shape[0]
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                propensities[i, j] = 0
                continue
            propensities[i, j] = c[i] * (r[i, j] * c[j] + r_tilde[i, j])
