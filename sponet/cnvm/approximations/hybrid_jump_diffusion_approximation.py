import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray


def sample_hybrid_jump_diffusion() -> tuple[NDArray, NDArray]:
    # TODO: execute many simulations
    pass


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
            r,
            r_tilde,
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
    r: NDArray,
    r_tilde: NDArray,
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
    r : NDArray
    r_tilde : NDArray
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
