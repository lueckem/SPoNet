import numpy as np
from scipy.integrate import solve_ivp
from ..parameters import CNVMParameters


def calc_rre_traj(
    params: CNVMParameters, c_0: np.ndarray, t_max: float, t_eval=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the RRE given by parameters, starting from c_0, up to time t_max.

    Solves the ODE using scipy's "solve_ivp".

    Parameters
    ----------
    params : CNVMParameters
    c_0 : np.ndarray
        Initial state, shape=(num_opinions,)
    t_max : float
        End time.
    t_eval : np.ndarray, optional
        Time points, at which the solution should be evaluated.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        1. timepoints, shape=(?,).
        2. c, shape=(?, num_opinions).
    """

    def rhs(t, c):
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

    sol = solve_ivp(rhs, (0, t_max), c_0, rtol=1e-8, atol=1e-8, t_eval=t_eval)
    return sol.t, sol.y.T
