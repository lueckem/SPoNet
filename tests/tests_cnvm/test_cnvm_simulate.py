import networkx as nx
import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray

import sponet.network_generator as ng
from sponet.cnvm.model import CNVM
from sponet.cnvm.parameters import CNVMParameters


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def x_init() -> NDArray:
    return np.ones(100)


@pytest.fixture
def r() -> NDArray:
    return np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])


@pytest.fixture
def r_tilde() -> NDArray:
    return np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])


@pytest.fixture
def params_complete(r, r_tilde) -> CNVMParameters:
    return CNVMParameters(
        num_agents=100,
        r=r,
        r_tilde=r_tilde,
    )


@pytest.fixture
def params_network(r, r_tilde) -> CNVMParameters:
    return CNVMParameters(
        network=nx.barabasi_albert_graph(100, 3),
        r=r,
        r_tilde=r_tilde,
        alpha=0,
    )


@pytest.fixture
def params_generator(r, r_tilde) -> CNVMParameters:
    return CNVMParameters(
        network_generator=ng.BarabasiAlbertGenerator(100, 3),
        r=r,
        r_tilde=r_tilde,
    )


@pytest.mark.parametrize(
    "params", ["params_complete", "params_network", "params_generator"]
)
def test_simulate_basic(params, x_init, rng, request):
    params = request.getfixturevalue(params)
    model = CNVM(params)
    t, x = model.simulate(10, x_init, rng=rng)
    assert t[0] == 0
    assert 9.9 < t[-1] < 10.1
    assert (x[0] == x_init).all()
    assert ((x == 0) | (x == 1) | (x == 2)).all()
    assert x.shape[1] == 100
    assert t.shape[0] == x.shape[0]


@pytest.mark.parametrize(
    "params", ["params_complete", "params_network", "params_generator"]
)
def test_simulate_linspace(params, x_init, rng, request):
    params = request.getfixturevalue(params)
    model = CNVM(params)
    t, x = model.simulate(10, x_init, t_eval=11, rng=rng)
    target_t = np.linspace(0, 10, 11)

    assert t.shape == (11,)
    assert np.allclose(t, target_t, atol=0.01)
    assert (x[0] == x_init).all()
    assert ((x == 0) | (x == 1) | (x == 2)).all()
    assert x.shape == (11, 100)


@pytest.mark.parametrize(
    "params", ["params_complete", "params_network", "params_generator"]
)
def test_simulate_teval(params, x_init, rng, request):
    params = request.getfixturevalue(params)
    model = CNVM(params)
    t_eval = [0, 3, 7, 9]
    t, x = model.simulate(10, x_init, t_eval=t_eval, rng=rng)

    assert t.shape == (4,)
    assert np.allclose(t, t_eval, atol=0.01)
    assert (x[0] == x_init).all()
    assert ((x == 0) | (x == 1) | (x == 2)).all()
    assert x.shape == (4, 100)


@pytest.mark.parametrize("params", ["params_complete", "params_network"])
def test_rng(params, request):
    params = request.getfixturevalue(params)
    model = CNVM(params)
    rng1 = np.random.default_rng(1234)
    t1, x1 = model.simulate(10, rng=rng1)

    model = CNVM(params)
    rng2 = np.random.default_rng(1234)
    t2, x2 = model.simulate(10, rng=rng2)

    assert np.allclose(t1, t2)
    assert np.allclose(x1, x2)


@pytest.mark.parametrize(
    "params", ["params_complete", "params_network", "params_generator"]
)
def test_output_concise(params, x_init, rng, request):
    # If t_eval is not specified, the output should only contain states that
    # have changed from one snapshot to the next
    params = request.getfixturevalue(params)
    model = CNVM(params)
    _, x = model.simulate(100, x_init, rng=rng)

    for i in range(x.shape[0] - 1):
        assert not np.allclose(x[i], x[i + 1])


@pytest.mark.parametrize(
    "params", ["params_complete", "params_network", "params_generator"]
)
def test_output_fill(params, x_init, rng, request):
    # If there are less transitions than requested,
    # the output should be filled with copies appropriately
    params = request.getfixturevalue(params)
    model = CNVM(params)
    t, x = model.simulate(1, x_init, t_eval=101, rng=rng)
    assert t.shape == (101,)
    assert x.shape == (101, 100)

    for i in range(x.shape[0] - 1):
        if (x[i] == x[i + 1]).all():  # there have to be duplicates
            break
        assert False


@pytest.mark.parametrize(
    "num_opinions,expected_dtype", [(2, np.uint8), (10, np.uint8), (257, np.uint16)]
)
def test_output_dtype(num_opinions, expected_dtype):
    # complete network
    params = CNVMParameters(
        num_opinions=num_opinions,
        num_agents=100,
        r=1,
        r_tilde=0.1,
    )
    model = CNVM(params)
    _, x = model.simulate(1)
    assert x.dtype == expected_dtype

    # network
    params = CNVMParameters(
        num_opinions=num_opinions,
        network=nx.barabasi_albert_graph(100, 3),
        r=1,
        r_tilde=0.1,
    )
    model = CNVM(params)
    _, x = model.simulate(1)
    assert x.dtype == expected_dtype


@pytest.fixture
def params_absorbing_complete() -> CNVMParameters:
    return CNVMParameters(num_agents=100, r=[[0, 1], [0, 0]], r_tilde=0)


@pytest.fixture
def params_absorbing_network() -> CNVMParameters:
    return CNVMParameters(
        network=nx.erdos_renyi_graph(100, 0.2), r=[[0, 1], [0, 0]], r_tilde=0
    )


@pytest.mark.parametrize(
    "params", ["params_absorbing_complete", "params_absorbing_network"]
)
def test_absorbing(params, request):
    params = request.getfixturevalue(params)
    model = CNVM(params)
    x_init = [0] * 50 + [1] * 50
    _, x = model.simulate(t_max=100, x_init=x_init, t_eval=101)
    # should reach absorbing state at about t=7

    # test reaches absorbing state
    idx_absorb = 0
    while idx_absorb < 101:
        if np.array_equal(x[idx_absorb, :], np.ones(100)):
            break
        idx_absorb += 1
    assert idx_absorb < 101

    # test stays in absorbing state
    for i in range(idx_absorb, 101):
        assert np.array_equal(x[i, :], np.ones(100))
