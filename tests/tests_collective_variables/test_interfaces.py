import networkx as nx
import pytest

import sponet.collective_variables as cvs

from .utils import assert_close


@pytest.fixture
def network():
    return nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4)])


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]], [[6], [4]]),
        ([0, 1, 0, 1, 1], [6]),
    ],
)
def test_default(network, x, expected):
    cv = cvs.Interfaces(network)
    assert_close(x, cv, expected)


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]], [[6 / 7], [4 / 7]]),
        ([0, 1, 0, 1, 1], [6 / 7]),
    ],
)
def test_normalize(network, x, expected):
    cv = cvs.Interfaces(network, normalize=True)
    assert_close(x, cv, expected)
