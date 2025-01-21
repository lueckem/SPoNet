from unittest import TestCase

import networkx as nx
import numpy as np

from sponet.utils import argmatch, calculate_neighbor_list, mask_subsequent_duplicates


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
