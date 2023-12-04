import numba
import numpy as np
from ..parameters import CNVMParameters


def sample_cle(
    params: CNVMParameters,
    initial_state: np.ndarray,
    max_time: float,
    num_time_steps: int,
    num_samples: int,
    saving_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample Chemical Langevin Equation (CLE) approximation for the CNVM.

    The Euler-Maruyama method is used to integrate the SDE.

    Parameters
    ----------
    params : CNVMParameters
    initial_state : np.ndarray
        Shape = (num_opinions,)
    max_time : float
    num_time_steps : int
        The step size of the integration is max_time / num_time_steps.
    num_samples : int
    saving_offset : int, optional
        Only return every saving_offset-th state to save memory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (t, c), t.shape=(num_time_steps + 1), c.shape = (num_samples, num_time_steps + 1, num_opinions).
        (If saving_offset > 1, the number of time steps will be smaller.)
    """
    return _numba_sample_cle(
        initial_state,
        max_time,
        num_time_steps,
        params.num_agents,
        params.r,
        params.r_tilde,
        num_samples,
        saving_offset,
    )


@numba.njit(parallel=True)
def _numba_sample_cle(
    initial_state: np.ndarray,
    max_time: float,
    num_time_steps: int,
    num_agents: int,
    r: np.ndarray,
    r_tilde: np.ndarray,
    num_samples: int,
    saving_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    dim = initial_state.shape[0]
    t = np.linspace(0, max_time, num_time_steps + 1)
    t = t[::saving_offset]
    x_out = np.zeros((num_samples, t.shape[0], dim))

    for i in numba.prange(num_samples):
        x = _numba_euler_maruyama(
            initial_state, max_time, num_time_steps, num_agents, r, r_tilde
        )
        x_out[i] = x[::saving_offset, :]

    return t, x_out


@numba.njit()
def _numba_euler_maruyama(
    initial_state: np.ndarray,
    max_time: float,
    num_time_steps: int,
    num_agents: int,
    r: np.ndarray,
    r_tilde: np.ndarray,
) -> np.ndarray:
    dim = initial_state.shape[0]
    x = np.zeros((num_time_steps + 1, dim))

    x[0] = np.copy(initial_state)
    delta_t = max_time / num_time_steps
    dim_diffusion = dim**2 - dim
    wiener_increments = np.random.normal(
        0, delta_t**0.5, (num_time_steps, dim_diffusion)
    )

    for i in range(num_time_steps):
        drift, diffusion = _drift_and_diffusion(x[i], r, r_tilde, num_agents)
        x[i + 1] = x[i] + drift * delta_t + diffusion @ wiener_increments[i]
        x[i + 1] = np.clip(x[i + 1], 0, 1)

    return x


@numba.njit()
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
