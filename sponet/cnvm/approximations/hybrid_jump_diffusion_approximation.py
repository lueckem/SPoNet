from numpy.typing import ArrayLike, NDArray
import numpy as np

from numba import njit


def sample_hybrid_jump_diffusion() -> tuple[NDArray, NDArray]:
    # TODO: execute many simulations
    pass


def _numba_jda():
    # TODO: execute one simulation
    pass


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
