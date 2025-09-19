import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

from sponet.cnvm.approximations.chemical_langevin_equation.boundary_processes import (
    BoundaryProcess,
    get_boundary_process_from_alias,
)

from sponet.cnvm.parameters import CNVMParameters


def sample_cle(
    params: CNVMParameters,
    initial_states: NDArray,
    t_max: float,
    num_samples: int,
    delta_t: float | None = None,
    t_eval: ArrayLike | None = None,
    boundary_process: str = "clipping",
) -> tuple[NDArray, NDArray]:
    """
    Sample Chemical Langevin Equation (CLE) approximation for the CNVM.

    The Euler-Maruyama method is used to integrate the SDE.

    Either `delta_t` or `t_eval` or both have to be provided.

    Parameters
    ----------
    params : CNVMParameters
    initial_states : NDArray
        Either shape = (num_opinions,) or (num_states, num_opinions)
    t_max : float
    num_samples : int
    delta_t : float, optional
        Step size.
    t_eval : ArrayLike, optional
        Array of time points where the solution should be saved,
        or number "n" in which case the solution is stored equidistantly at "n" time points.
    boundary_process : str
        Kind of process used to deal with the approximation leaving the simplex boundary.
        Possible values: "clipping", "jump", "normal-reflection"
        Defaults to "clipping".

    Returns
    -------
    tuple[NDArray, NDArray]
        (t, c),
        t.shape=(num_timesteps),
        c.shape = (num_states, num_samples, num_timesteps, num_opinions), or c.shape = (num_samples, num_timesteps, num_opinions) if a single initial state was given.
        (If saving_offset > 1, the number of time steps will be smaller.)
    """
    delta_t, t_eval = _sanitize_delta_t_and_t_eval(delta_t, t_eval, t_max)

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

    boundary_process = get_boundary_process_from_alias(boundary_process)

    for i in range(num_states):
        t, c[i] = _numba_sample_cle(
            initial_states[i],
            delta_t,
            t_eval,
            params.num_agents,
            params.r,
            params.r_tilde,
            num_samples,
            boundary_process,
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
        if isinstance(t_eval, float):
            raise ValueError("t_eval has to be an array of time points or an int.")

        if isinstance(t_eval, int):
            t_eval = np.linspace(0, max_time, t_eval)

        t_eval = np.array(t_eval)
        if np.min(t_eval) < 0:
            raise ValueError("The times in t_eval have to be >= 0.")

        diffs = np.diff(t_eval)
        if np.min(diffs) <= 0:
            raise ValueError("The times in t_eval have to be increasing.")

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
def _numba_sample_cle(
    initial_state: NDArray,
    delta_t: float,
    t_eval: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
    num_samples: int,
    boundary_process: BoundaryProcess,
) -> tuple[NDArray, NDArray]:
    dim = initial_state.shape[0]
    x_out = np.zeros((num_samples, t_eval.shape[0], dim))

    for i in prange(num_samples):
        x_out[i] = _numba_euler_maruyama(
            initial_state, delta_t, t_eval, num_agents, r, r_tilde, boundary_process
        )

    return t_eval, x_out


@njit()
def _numba_euler_maruyama(
    initial_state: np.ndarray,
    delta_t: float,
    t_eval: NDArray,
    num_agents: int,
    r: NDArray,
    r_tilde: NDArray,
    boundary_process: BoundaryProcess,
) -> NDArray:
    dim = initial_state.shape[0]
    dim_diffusion = dim**2 - dim

    x_store = np.zeros((t_eval.shape[0], dim))
    x_store[0] = initial_state

    x = np.copy(initial_state)
    t = 0.0
    next_store_index = 1
    next_t_store = t_eval[next_store_index]
    while True:
        if t + delta_t >= next_t_store:
            this_delta_t = next_t_store - t
            store = True
        else:
            this_delta_t = delta_t
            store = False

        # NOTE: possible performance improvement:
        # Maybe drift, diffusion, wiener_increments allocate at each step.
        # Make sure that the new ones overwrite the old ones in place.
        drift, diffusion = _drift_and_diffusion(x, r, r_tilde, num_agents)
        wiener_increments = np.random.normal(0, this_delta_t**0.5, dim_diffusion)
        x_new = x + drift * delta_t + diffusion @ wiener_increments

        # Check if trajectory left boundary
        if (x_new <= 0).any():
            x_store, t, x_new, next_store_index, changed_time = boundary_process(
                t_eval,
                x_store,
                t,
                t + this_delta_t,
                x,
                x_new,
                next_store_index,
                num_agents,
                r,
                r_tilde,
            )
            # If boundary process advanced time, it either advanced past next_t_store or it stopped before it.
            # In both cases a new euler-maruyama step has to be computed before storing any value.
            if changed_time:
                store = False
        else:
            t += this_delta_t

        x[:] = x_new

        if store:
            x_store[next_store_index] = x
            next_store_index += 1
            if next_store_index >= t_eval.shape[0]:
                break
            next_t_store = t_eval[next_store_index]

    return x_store


@njit()
def _drift_and_diffusion(c, r, r_tilde, num_agents):
    num_o = c.shape[0]
    drift = np.zeros(num_o)
    diffusion = np.zeros((num_o, num_o**2 - num_o))

    i = 0
    for m in range(num_o):
        for n in range(num_o):
            if n == m:
                continue

            state_change = np.zeros(num_o)
            state_change[m] = -1
            state_change[n] = 1
            prop = c[m] * (r[m, n] * c[n] + r_tilde[m, n])
            drift += prop * state_change
            diffusion[:, i] = (prop / num_agents) ** 0.5 * state_change
            i += 1

    return drift, diffusion
