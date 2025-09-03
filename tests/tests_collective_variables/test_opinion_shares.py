import networkx as nx
import numpy as np
import pytest

import sponet.collective_variables as cvs

from .utils import assert_close


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 0, 1, 2, 2, 1, 1, 1, 0]], [[3, 5, 2]]),
        ([0, 1, 0, 1, 2, 2, 1, 1, 1, 0], [3, 5, 2]),
        ([[0, 1, 0, 1, 0, 0, 0], [1, 1, 1, 2, 1, 2, 2]], [[5, 2, 0], [0, 4, 3]]),
        ([1, 1, 1, 2, 1, 2, 2], [0, 4, 3]),
    ],
)
def test_default(x, expected):
    cv = cvs.OpinionShares(3)
    assert_close(x, cv, expected)


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 0, 1, 2, 2, 1, 1, 1, 0]], [[0.3, 0.5, 0.2]]),
        ([0, 1, 0, 1, 2, 2, 1, 1, 1, 0], [0.3, 0.5, 0.2]),
        (
            [[0, 1, 0, 1, 0, 0, 0], [1, 1, 1, 2, 1, 2, 2]],
            [[5 / 7, 2 / 7, 0], [0, 4 / 7, 3 / 7]],
        ),
    ],
)
def test_normalize(x, expected):
    cv = cvs.OpinionShares(3, normalize=True)
    assert_close(x, cv, expected)


@pytest.mark.parametrize(
    "x,normalize,expected",
    [
        ([[0, 1, 0, 2, 1]], False, [[1.5, 1, 0]]),
        ([0, 1, 0, 2, 1], False, [1.5, 1, 0]),
        ([[0, 0, 1, 1, 2]], True, [[1 / 2.5, 0.5 / 2.5, 1 / 2.5]]),
        ([0, 0, 1, 1, 2], True, [1 / 2.5, 0.5 / 2.5, 1 / 2.5]),
    ],
)
def test_weights(x, normalize, expected):
    weights = [1, 0, 0.5, 0, 1]
    cv = cvs.OpinionShares(3, weights=weights, normalize=normalize)
    assert_close(x, cv, expected)


@pytest.mark.parametrize(
    "idx,expected",
    [
        (1, [5]),
        ([0, 1], [3, 5]),
        (np.array([0, 1]), [3, 5]),
        ([2, 0, 1], [2, 3, 5]),
    ],
)
def test_idx(idx, expected):
    x = np.array([0, 1, 0, 1, 2, 2, 1, 1, 1, 0])
    cv = cvs.OpinionShares(3, idx_to_return=idx)
    assert_close(x, cv, expected)


def test_degree_weighted():
    network = nx.star_graph(9)
    cv = cvs.DegreeWeightedOpinionShares(3, network)
    x = [0, 1, 0, 1, 2, 2, 1, 1, 1, 0]
    expected = [11, 5, 2]
    assert_close(x, cv, expected)
