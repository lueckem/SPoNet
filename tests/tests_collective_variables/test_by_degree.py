import networkx as nx
import numpy as np
import pytest

import sponet.collective_variables as cvs

from .utils import assert_close


@pytest.fixture
def network():
    return nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4)])


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]], [[0, 2, 1, 2, 0, 0], [2, 1, 0, 0, 2, 0]]),
        ([0, 1, 0, 2, 1], [0, 2, 1, 2, 0, 0]),
    ],
)
def test_default(network, x, expected):
    cv = cvs.OpinionSharesByDegree(3, network)
    assert_close(x, cv, expected)


@pytest.mark.parametrize(
    "x,expected",
    [
        (
            [[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]],
            [[0, 2 / 3, 1 / 3, 1, 0, 0], [2 / 3, 1 / 3, 0, 0, 1, 0]],
        ),
        ([0, 1, 0, 2, 1], [0, 2 / 3, 1 / 3, 1, 0, 0]),
    ],
)
def test_normalize(network, x, expected):
    cv = cvs.OpinionSharesByDegree(3, network, normalize=True)
    assert_close(x, cv, expected)


@pytest.mark.parametrize(
    "idx,expected",
    [
        (0, [[0, 2], [2, 0]]),
        ([2, 0], [[1, 0, 0, 2], [0, 2, 0, 0]]),
        (np.array([2, 0]), [[1, 0, 0, 2], [0, 2, 0, 0]]),
    ],
)
def test_idx(network, idx, expected):
    x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
    cv = cvs.OpinionSharesByDegree(3, network, idx_to_return=idx)
    assert_close(x, cv, expected)
