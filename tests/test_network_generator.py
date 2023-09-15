from unittest import TestCase
import numpy as np
import networkx as nx

import sponet.network_generator as ng


class TestStochasticBlockGenerator(TestCase):
    def test_shape(self):
        p_matrix = np.array([[1, 0.5, 0], [0.5, 0, 1], [0, 1, 0]])
        num_agents = 90
        sbm = ng.StochasticBlockGenerator(num_agents, p_matrix)
        network = sbm()
        adj_matrix = nx.to_scipy_sparse_array(network).todense()
        self.assertEqual(adj_matrix.shape, (90, 90))
        self.assertTrue(np.allclose(adj_matrix, adj_matrix.T))
        self.assertTrue(np.allclose(np.diag(adj_matrix), np.zeros(90)))
        self.assertTrue(np.allclose(adj_matrix[:30, 60:], np.zeros((30, 30))))
        self.assertTrue(np.allclose(adj_matrix[30:60, 60:], np.ones((30, 30))))
