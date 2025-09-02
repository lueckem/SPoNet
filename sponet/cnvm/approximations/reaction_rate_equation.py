from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..parameters import CNVMParameters


def calc_rre_traj(
    params: CNVMParameters,
    initial_states: NDArray,
    t_max: float,
    t_eval: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Solve the RRE given by parameters, starting from c_0, up to time t_max.

    Solves the ODE using scipy's "solve_ivp".

    Parameters
    ----------
    params : CNVMParameters
    initial_states : NDArray
        Either shape = (num_opinions,) or (num_states, num_opinions)
    t_max : float
        End time.
    t_eval : NDArray, optional
        Time points, at which the solution should be evaluated.

    Returns
    -------
    tuple[NDArray, NDArray]
        (t, c),
        t.shape=(num_time_steps,),
        c.shape = (num_states, num_time_steps, num_opinions), or c.shape = (num_time_steps, num_opinions) if a single initial state was given.
    """

    def rhs(_, c):
        out = np.zeros_like(c)
        for m in range(params.num_opinions):
            for n in range(params.num_opinions):
                if n == m:
                    continue

                state_change = np.zeros_like(c)
                state_change[m] = -1
                state_change[n] = 1

                prop = c[m] * (
                    params.r_imit * params.prob_imit[m, n] * c[n]
                    + params.r_noise * params.prob_noise[m, n] / params.num_opinions
                )

                out += prop * state_change
        return out

    if initial_states.ndim == 1:
        sol = solve_ivp(
            rhs, (0, t_max), initial_states, rtol=1e-8, atol=1e-8, t_eval=t_eval
        )
        return sol.t, sol.y.T

    c = []
    for initial_state in initial_states:
        sol = solve_ivp(
            rhs, (0, t_max), initial_state, rtol=1e-8, atol=1e-8, t_eval=t_eval
        )
        t = sol.t
        c.append(sol.y.T)
    return t, np.array(c)  # type: ignore


def calc_modified_rre_traj(
    params: CNVMParameters,
    initial_states: NDArray,
    t_max: float,
    alpha: float = 1.0,
    t_eval: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Solve the RRE with modified parameters, starting from c_0, up to time t_max.

    The parameters are modified by multiplying the imitation rates `r` with the factor `alpha`.
    For instance, if alpha < 1, this effectively slows the dynamics.

    Parameters
    ----------
    params : CNVMParameters
    initial_states : NDArray
        Either shape = (num_opinions,) or (num_states, num_opinions)
    t_max : float
        End time.
    alpha : float
        Factor for modification of imitation rates.
    t_eval : NDArray, optional
        Time points, at which the solution should be evaluated.

    Returns
    -------
    tuple[NDArray, NDArray]
        (t, c),
        t.shape=(num_time_steps,),
        c.shape = (num_states, num_time_steps, num_opinions), or c.shape = (num_time_steps, num_opinions) if a single initial state was given.
    """
    modified_params = deepcopy(params)
    modified_params.change_rates(r=alpha * params.r)
    return calc_rre_traj(modified_params, initial_states, t_max, t_eval)
