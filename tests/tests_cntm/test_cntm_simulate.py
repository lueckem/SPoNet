import networkx as nx
import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray

from sponet.cntm.model import CNTM
from sponet.cntm.parameters import CNTMParameters


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng(123)


@pytest.fixture
def x_init() -> NDArray:
    return np.array([1] * 50 + [0] * 50)


@pytest.fixture
def params() -> CNTMParameters:
    return CNTMParameters(
        network=nx.erdos_renyi_graph(100, 0.1, seed=123),
        r=1,
        r_tilde=0.2,
        threshold_01=0.5,
        threshold_10=0.5,
    )


def test_simulate_basic(params, x_init, rng):
    model = CNTM(params)
    t, x = model.simulate(10, x_init, rng=rng)
    assert t[0] == 0
    assert 9.9 < t[-1] < 10.1
    assert (x[0] == x_init).all()
    assert ((x == 0) | (x == 1)).all()
    assert x.shape[1] == 100
    assert t.shape[0] == x.shape[0]


def test_simulate_linspace(params, x_init, rng):
    model = CNTM(params)
    t, x = model.simulate(10, x_init, t_eval=11, rng=rng)
    target_t = np.linspace(0, 10, 11)

    assert t.shape == (11,)
    assert np.allclose(t, target_t, atol=0.01)
    assert (x[0] == x_init).all()
    assert ((x == 0) | (x == 1)).all()
    assert x.shape == (11, 100)


def test_simulate_teval(params, x_init, rng):
    model = CNTM(params)
    t_eval = [0, 3, 7, 9]
    t, x = model.simulate(10, x_init, t_eval=t_eval, rng=rng)

    assert t.shape == (4,)
    assert np.allclose(t, t_eval, atol=0.01)
    assert (x[0] == x_init).all()
    assert ((x == 0) | (x == 1)).all()
    assert x.shape == (4, 100)


def test_rng(params):
    model = CNTM(params)
    rng1 = np.random.default_rng(1234)
    t1, x1 = model.simulate(10, rng=rng1)

    rng2 = np.random.default_rng(1234)
    t2, x2 = model.simulate(10, rng=rng2)

    assert np.allclose(t1, t2)
    assert np.allclose(x1, x2)


def test_output_concise(params, x_init, rng):
    # If t_eval is not specified, the output should only contain states that
    # have changed from one snapshot to the next
    model = CNTM(params)
    _, x = model.simulate(100, x_init, rng=rng)

    for i in range(x.shape[0] - 1):
        assert not np.allclose(x[i], x[i + 1])


def test_output_fill(params, x_init, rng):
    # If there are less transitions than requested,
    # the output should be filled with copies appropriately
    model = CNTM(params)
    t, x = model.simulate(1, x_init, t_eval=101, rng=rng)
    assert t.shape == (101,)
    assert x.shape == (101, 100)

    for i in range(x.shape[0] - 1):
        if (x[i] == x[i + 1]).all():  # there have to be duplicates
            break
        assert False
