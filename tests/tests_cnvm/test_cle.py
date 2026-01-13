import numpy as np
import pytest

from sponet.cnvm.approximations.chemical_langevin_equation.chemical_langevin_equation import (
    sample_cle,
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
    delta_t = 0.1
    t_eval = 101
    t, c = sample_cle(params, initial_state, max_time, num_samples, delta_t, t_eval)
    assert c.shape == c_shape
    assert t.shape == (101,)


@pytest.mark.parametrize(
    "delta_t,t_eval,expected_t",
    [
        (0.1, np.linspace(0, 100, 101), np.linspace(0, 100, 101)),
        (0.1, [0, 2.5, 76, 99.4], np.array([0, 2.5, 76, 99.4])),
        (0.1, 101, np.linspace(0, 100, 101)),
        (0.1, None, np.linspace(0, 100, 1001)),
        (None, np.linspace(0, 100, 101), np.linspace(0, 100, 101)),
        (None, 101, np.linspace(0, 100, 101)),
    ],
)
def test_t_eval_delta_t(params, delta_t, t_eval, expected_t):
    initial_state = [0.1, 0.7, 0.2]
    t, c = sample_cle(params, initial_state, 100, 25, delta_t, t_eval)
    assert np.allclose(t, expected_t)
    assert c.shape[1] == t.shape[0]
    assert np.allclose(c[:, 0, :], initial_state)
