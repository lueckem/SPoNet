from numpy.typing import ArrayLike, NDArray


def sample_hybrid_jump_diffusion() -> tuple[NDArray, NDArray]:
    # TODO: execute many simulations
    pass


def _numba_jda():
    # TODO: execute one simulation
    pass


def _numba_update_channels(
    c: NDArray,
    jump_channels: NDArray,
    jump_thresholds: NDArray,
    jump_integrated_times: NDArray,
    switch_propensity_threshold: float,
    switch_boundary_thresholds: NDArray,
):
    pass


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
