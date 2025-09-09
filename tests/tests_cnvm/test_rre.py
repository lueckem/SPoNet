import numpy as np
import pytest

from sponet.cnvm.approximations.reaction_rate_equation import calc_rre_traj
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
        ([0.1, 0.9, 0], (101, 3)),
        ([[0.2, 0.5, 0.3], [0.1, 0.9, 0]], (2, 101, 3)),
    ],
)
def test_shape(params, initial_state, c_shape):
    t, c = calc_rre_traj(params, np.array(initial_state), 100, 101)
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
def test_t_eval(params, t_eval, expected_t):
    initial_state = np.array([0.1, 0.7, 0.2])
    t, c = calc_rre_traj(params, initial_state, 100, t_eval)
    assert np.allclose(t, expected_t)
    assert c.shape[0] == t.shape[0]
    assert np.allclose(c[0, :], initial_state)
