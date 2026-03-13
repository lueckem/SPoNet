import numpy as np

from sponet import (
    CNVMParameters,
    calc_rre_traj,
    sample_cle,
    sample_many_runs,
    sample_stochastic_approximation,
)
from sponet.collective_variables import OpinionShares


def test_agreement_of_approximations():
    num_opinions = 3
    num_agents = 100000
    r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
    r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

    params = CNVMParameters(
        num_opinions=num_opinions,
        num_agents=num_agents,
        r=r,
        r_tilde=r_tilde,
    )
    cv = OpinionShares(num_opinions, True)
    t_max = 30
    num_time_steps = 31
    num_samples = 100

    initial_c = np.array([0.9, 0.1, 0.0])
    initial_x = np.array([0] * 90000 + [1] * 10000 + [2] * 0)

    t, c = sample_many_runs(
        params,
        initial_x,
        t_max,
        num_time_steps,
        num_samples,
        n_jobs=-1,
        collective_variable=cv,
    )
    t_cle, c_cle = sample_cle(
        params,
        initial_c,
        t_max,
        num_samples,
        t_eval=num_time_steps,
        delta_t=0.01,
    )
    t_sa, c_sa = sample_stochastic_approximation(
        params, initial_c, t_max, num_samples, num_time_steps
    )
    t_rre, c_rre = calc_rre_traj(params, initial_c, t_max, num_time_steps)

    mean_c = np.mean(c, axis=0)
    mean_c_cle = np.mean(c_cle, axis=0)
    mean_c_sa = np.mean(c_sa, axis=0)

    assert np.allclose(t, t_cle)
    assert np.allclose(t, t_sa)
    assert np.allclose(t, t_rre)

    assert np.mean((mean_c - mean_c_cle) ** 2) < 1e-3
    assert np.mean((mean_c - mean_c_sa) ** 2) < 1e-3
    assert np.mean((mean_c - c_rre) ** 2) < 1e-3
