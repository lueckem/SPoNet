import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

from sponet import (
    CNVMParameters,
    calc_pair_approximation_traj,
    calc_rre_traj,
    sample_cle,
    sample_many_runs,
    sample_stochastic_approximation,
)
from sponet.collective_variables import OpinionShares


@pytest.fixture
def params() -> CNVMParameters:
    num_opinions = 3
    num_agents = 100
    r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
    r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

    return CNVMParameters(
        num_opinions=num_opinions,
        num_agents=num_agents,
        r=r,
        r_tilde=r_tilde,
    )


@pytest.fixture
def t_max() -> int:
    return 100


@pytest.fixture
def t_eval() -> NDArray:
    return np.linspace(0, 100, 1001)  # type: ignore


@pytest.fixture
def num_time_steps() -> int:
    return 1000


@pytest.fixture
def num_samples() -> int:
    return 20


@pytest.mark.parametrize(
    "initial_states,c_shape",
    [
        ([0.9, 0.1, 0.0], (1001, 3)),
        ([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5]], (2, 1001, 3)),
    ],
)
def test_calc_rre(params, t_max, t_eval, initial_states, c_shape):
    t, c = calc_rre_traj(params, np.array(initial_states), t_max, t_eval)
    assert np.allclose(t, t_eval)
    assert c.shape == c_shape
    if c.ndim == 2:
        assert np.allclose(c[0], initial_states)
    else:
        assert np.allclose(c[:, 0], initial_states)


@pytest.mark.parametrize(
    "initial_states,c_shape",
    [
        ([0.9, 0.1, 0.0], (20, 1001, 3)),
        ([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5]], (2, 20, 1001, 3)),
    ],
)
def test_sample_cle(
    params, t_max, num_time_steps, num_samples, t_eval, initial_states, c_shape
):
    delta_t = 1e-3
    t, c = sample_cle(
        params,
        np.array(initial_states),
        t_max,
        num_samples,
        t_eval=num_time_steps + 1,
        delta_t=delta_t,
    )
    assert np.allclose(t, t_eval)
    assert c.shape == c_shape
    if c.ndim == 3:
        assert np.allclose(c[:, 0], initial_states)
    else:
        for i in range(c.shape[1]):
            assert np.allclose(c[:, i, 0], initial_states)


@pytest.mark.parametrize(
    "initial_states,c_shape",
    [
        ([0.9, 0.1, 0.0], (20, 1001, 3)),
        ([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5]], (2, 20, 1001, 3)),
    ],
)
def test_sample_stochastic_approx(
    params, t_max, num_time_steps, num_samples, t_eval, initial_states, c_shape
):
    t, c = sample_stochastic_approximation(
        params, np.array(initial_states), t_max, num_time_steps, num_samples
    )
    assert np.allclose(t, t_eval)
    assert c.shape == c_shape
    if c.ndim == 3:
        assert np.allclose(c[:, 0], initial_states)
    else:
        for i in range(c.shape[1]):
            assert np.allclose(c[:, i, 0], initial_states)


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
    num_time_steps = 30
    num_samples = 100

    initial_c = np.array([0.9, 0.1, 0.0])
    initial_x = np.array([0] * 90000 + [1] * 10000 + [2] * 0)

    t, c = sample_many_runs(
        params,
        initial_x,
        t_max,
        num_time_steps + 1,
        num_samples,
        n_jobs=-1,
        collective_variable=cv,
    )
    t_cle, c_cle = sample_cle(
        params,
        initial_c,
        t_max,
        num_samples,
        t_eval=num_time_steps + 1,
        delta_t=0.01,
    )
    t_sa, c_sa = sample_stochastic_approximation(
        params, initial_c, t_max, num_time_steps, num_samples
    )
    t_rre, c_rre = calc_rre_traj(
        params, initial_c, t_max, np.linspace(0, t_max, num_time_steps + 1)
    )

    mean_c = np.mean(c, axis=0)
    mean_c_cle = np.mean(c_cle, axis=0)
    mean_c_sa = np.mean(c_sa, axis=0)

    assert np.allclose(t, t_cle)
    assert np.allclose(t, t_sa)
    assert np.allclose(t, t_rre)

    assert np.mean((mean_c - mean_c_cle) ** 2) < 1e-3
    assert np.mean((mean_c - mean_c_sa) ** 2) < 1e-3
    assert np.mean((mean_c - c_rre) ** 2) < 1e-3


def test_calc_pair_approximation_traj():
    from sponet.collective_variables import Interfaces

    num_opinions = 2
    num_agents = 1000
    r = np.array([[0, 1], [1.1, 0]])
    r_tilde = 0.01
    network = nx.barabasi_albert_graph(num_agents, 3)

    params = CNVMParameters(
        num_opinions=num_opinions,
        network=network,
        r=r,
        r_tilde=r_tilde,
    )

    t_max = 100
    num_time_steps = 10000
    x_0 = np.zeros(num_agents)
    x_0[:100] = 1
    c_0 = 0.1
    s_0 = 0.5 * Interfaces(network, True)(np.array([x_0]))[0, 0]
    mean_degree = np.mean([d for _, d in network.degree()])  # type: ignore

    t, c_pa = calc_pair_approximation_traj(
        params,
        c_0,
        s_0,
        mean_degree,  # type: ignore
        t_max,
        t_eval=np.linspace(0, t_max, num_time_steps + 1),
    )

    assert c_pa.shape == (10001, 2)
    assert c_pa[0, 0] == c_0
    assert c_pa[0, 1] == s_0
    assert np.allclose(t, np.linspace(0, t_max, num_time_steps + 1))
