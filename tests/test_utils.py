from unittest import TestCase

import networkx as nx
import numpy as np
import pytest

from sponet.utils import (
    argmatch,
    calculate_neighbor_list,
    counts_from_shares,
    mask_subsequent_duplicates,
)


class TestArgmatch(TestCase):
    def test_1(self):
        t = np.array([1, 2, 3, 4, 5, 6, 7])
        t_ref = np.array([1.8, 2, 4.4, 6.7, 6.9])
        ind = argmatch(t_ref, t)
        self.assertTrue(np.allclose(ind, [1, 1, 3, 6, 6]))

    def test_2(self):
        t = np.array([1, 2, 3, 4, 5])
        t_ref = np.array([0.1, 1.8, 2, 2.5, 4.4, 6.7, 7.7])
        ind = argmatch(t_ref, t)
        self.assertTrue(np.allclose(ind, [0, 1, 1, 2, 3, 4, 4]))


class TestMaskSubsequentDuplicates(TestCase):
    def test_1d(self):
        x = np.array([1, 1, 2, 2, 2, 3, 1, 3, 3], dtype=int)
        mask = mask_subsequent_duplicates(x)
        self.assertTrue(np.all(x[mask] == [1, 2, 3, 1, 3]))

    def test_2d(self):
        x = np.array([[1, 1], [1, 1], [2, 2], [2, 1], [2, 1], [1, 1]], dtype=int)
        mask = mask_subsequent_duplicates(x)
        self.assertTrue(np.all(x[mask] == [[1, 1], [2, 2], [2, 1], [1, 1]]))

    def test_exception(self):
        with self.assertRaises(ValueError):
            mask_subsequent_duplicates(np.zeros((3, 3, 3)))


class TestNeighborList(TestCase):
    def test_complete(self):
        network = nx.complete_graph(3)
        expected = [
            np.array([1, 2]),
            np.array([0, 2]),
            np.array([0, 1]),
        ]
        neighbor_list = calculate_neighbor_list(network)

        self.assertEqual(len(neighbor_list), 3)
        for i in range(0, 3):
            self.assertCountEqual(neighbor_list[i], expected[i])

    def test_star(self):
        network = nx.star_graph(3)
        expected = [
            np.array([1, 2, 3]),
            np.array([0]),
            np.array([0]),
            np.array([0]),
        ]
        neighbor_list = calculate_neighbor_list(network)

        self.assertEqual(len(neighbor_list), 4)
        for i in range(0, 4):
            self.assertCountEqual(neighbor_list[i], expected[i])


@pytest.mark.parametrize(
    "shares,num_agents,counts",
    [
        ([[0.2, 0.3, 0.5], [0.209, 0.309, 0.482]], 100, [[20, 30, 50], [21, 31, 48]]),
        ([[0.206, 0.306, 0.488]], 100, [[21, 30, 49]]),
        ([0.204, 0.304, 0.492], 100, [21, 30, 49]),
        ([0.555, 0.445, 0], 100, [56, 44, 0]),
        ([0.554, 0.446, 0], 100, [55, 45, 0]),
        ([0.554, 0.446, 0], 10, [6, 4, 0]),
        ([0.155, 0.155, 0.155, 0.155, 0.380], 100, [16, 16, 15, 15, 38]),
        ([0.055] * 18 + [0.010], 100, [6] * 9 + [5] * 9 + [1]),
        ([0.025] * 40 + [0], 100, [3] * 20 + [2] * 20 + [0]),
    ],
)
def test_counts_from_shares(shares, num_agents, counts):
    assert (counts_from_shares(shares, num_agents) == np.array(counts)).all()
