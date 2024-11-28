import numpy as np

from sponet.cntm.model import CNTM
from sponet.cntm.parameters import CNTMParameters
from sponet.cnvm.model import CNVM
from sponet.cnvm.parameters import CNVMParameters
from sponet.collective_variables import CollectiveVariable

from .parameters import Parameters


def estimate_steady_state(
    params: Parameters, col_var: CollectiveVariable, tol_abs: float = 0.01
) -> np.ndarray:
    """
    Estimate steady state of collective variable via a long simulation.

    It is designed to estimate within the absolute tolerance with 95% confidence,
    although this assumes stochastically independent samples, which is not the case.

    Parameters
    ----------
    params : Parameters
        CNVMParameters or CNTMParameters
    col_var: CollectiveVariable
    tol_abs : float, optional
        absolute tolerance

    Returns
    -------
    np.ndarray
    """
    if isinstance(params, CNVMParameters):
        delta_t = 1.0 / (
            np.max(params.r) + params.num_opinions * np.max(params.r_tilde)
        )
        model = CNVM(params)
    elif isinstance(params, CNTMParameters):
        delta_t = 1.0 / (params.r + params.r_tilde)
        model = CNTM(params)
    else:
        raise ValueError("Parameters not valid.")

    delta_t = float(delta_t)
    print(f"delta_t = {delta_t}")

    # transient phase: integrate until steady state
    print("Integrating through transient phase...")
    _, x = model.simulate(1000 * delta_t, len_output=2)

    # sampling phase: integrate until tol
    remaining_samples = 1000
    c = col_var(x)[1:]
    mean_c = c[-1]

    while remaining_samples > 0:
        print(f"Current esimate: {mean_c}")
        print(f"Acquiring {remaining_samples} more samples...")

        _, x = model.simulate(
            remaining_samples * delta_t, x[-1], len_output=remaining_samples + 1
        )

        c_new = col_var(x)[1:]
        c = np.concatenate([c, c_new], axis=0)
        mean_c = np.mean(c, axis=0)
        var_c = np.var(c, axis=0)
        total_samples = int(4 * np.max(var_c) / tol_abs**2)
        remaining_samples = total_samples - c.shape[0]

    print(f"Final estimate: {mean_c}")
    return mean_c
