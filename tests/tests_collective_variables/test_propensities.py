import networkx as nx
import pytest

import sponet.collective_variables as cvs
from sponet.cnvm.parameters import CNVMParameters

from .utils import assert_close


@pytest.fixture
def network():
    return nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4)])


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]], [[1.1, 1.4], [1.7, 1.8]]),
        ([0, 1, 1, 1, 1], [1.1, 1.4]),
    ],
)
def test_complete_network(x, expected):
    params = CNVMParameters(num_opinions=2, num_agents=5, r=1, r_tilde=0.1)
    cv = cvs.Propensities(params)
    assert_close(x, cv, expected)


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]], [[6.2, 6.3], [4.2, 4.3]]),
        ([0, 1, 0, 1, 1], [6.2, 6.3]),
    ],
)
def test_network(network, x, expected):
    params = CNVMParameters(num_opinions=2, network=network, r=1, r_tilde=0.1, alpha=0)
    cv = cvs.Propensities(params)
    assert_close(x, cv, expected)


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]], [[6.2 / 5, 6.3 / 5], [4.2 / 5, 4.3 / 5]]),
        ([0, 1, 0, 1, 1], [6.2 / 5, 6.3 / 5]),
    ],
)
def test_network_normalized(network, x, expected):
    params = CNVMParameters(num_opinions=2, network=network, r=1, r_tilde=0.1, alpha=0)
    cv = cvs.Propensities(params, normalize=True)
    assert_close(x, cv, expected)
