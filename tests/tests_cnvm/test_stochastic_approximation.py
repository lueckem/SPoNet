import numpy as np
import pytest

from sponet.cnvm.approximations.stochastic_approximation import (
    sample_stochastic_approximation,
)
from sponet.cnvm.parameters import CNVMParameters


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


@pytest.mark.parametrize(
    "initial_state,c_shape",
    [
        ([0.1, 0.9, 0], (25, 101, 3)),
        ([[0.2, 0.5, 0.3], [0.1, 0.9, 0]], (2, 25, 101, 3)),
    ],
)
def test_shape(params, initial_state, c_shape):
    max_time = 100
    num_samples = 25
    t_eval = 101
    t, c = sample_stochastic_approximation(
        params, initial_state, max_time, num_samples, t_eval
    )
    assert c.shape == c_shape
    assert t.shape == (101,)


@pytest.mark.parametrize(
    "t_eval,expected_t",
    [
        (np.linspace(0, 100, 101), np.linspace(0, 100, 101)),
        ([0, 2.5, 76, 99.4], np.array([0, 2.5, 76, 99.4])),
        (101, np.linspace(0, 100, 101)),
    ],
)
def test_t_eval_deta_t(params, t_eval, expected_t):
    initial_state = [0.1, 0.7, 0.2]
    t, c = sample_stochastic_approximation(params, initial_state, 100, 25, t_eval)
    assert np.allclose(t, expected_t)
    assert c.shape[1] == t.shape[0]
    assert np.allclose(c[:, 0, :], initial_state)
