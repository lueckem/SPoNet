import numpy as np
from scipy.integrate import solve_ivp

from ..parameters import CNVMParameters


def calc_pair_approximation_traj(
    params: CNVMParameters,
    c_0: float,
    s_0: float,
    d: float,
    t_max: float,
    t_eval=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For a CNVM with 2 opinions, calculate the pair approximation.

    The opinions are referred to as "0" and "1".
    The pair approximation is a system of two ODEs.
    The first ODE describes the evolution of the share of nodes with state 1, which is called "c".
    The second ODE describes the evolution of the variable "s",
    which can be interpreted has 0.5 times the share of active edges.
    So in this package, s can be computed as s = 0.5 * sponet.collective_variables.Interfaces.

    Parameters
    ----------
    params : CNVMParameters
    c_0 : float
        Initial state of c.
    s_0 : float
        Initial state of s.
    t_max : float
        End time.
    t_eval : np.ndarray, optional
        Time points, at which the solution should be evaluated.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        1. timepoints, shape=(?,).
        2. c, shape=(?, 2).
    """

    assert params.num_opinions == 2

    def rhs(_, c_pa):
        # assert isinstance(params.r, np.ndarray)
        # assert isinstance(params.r_tilde, np.ndarray)

        c, s = c_pa

        dc = (
            s * (params.r[0, 1] - params.r[1, 0])
            - c * (params.r_tilde[0, 1] + params.r_tilde[1, 0])
            + params.r_tilde[0, 1]
        )

        ds = -2 * (d - 1) / d * s * s / (1 - c) * params.r[0, 1]
        ds += -2 * (d - 1) / d * s * s / c * params.r[1, 0]
        ds += s * (
            (d - 2) / d * (params.r[0, 1] + params.r[1, 0])
            - 2 * params.r_tilde[0, 1]
            - 2 * params.r_tilde[1, 0]
        )
        ds += c * (params.r_tilde[1, 0] - params.r_tilde[0, 1])
        ds += params.r_tilde[0, 1]

        return np.array([dc, ds])

    sol = solve_ivp(
        rhs, (0, t_max), np.array([c_0, s_0]), rtol=1e-8, atol=1e-8, t_eval=t_eval
    )
    return sol.t, sol.y.T
