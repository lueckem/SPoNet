from unittest import TestCase

import networkx as nx
import numpy as np

import sponet.collective_variables as cvs
from sponet.cnvm.parameters import CNVMParameters


class TestOpinionSharesByDegree(TestCase):
    def setUp(self):
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4)]
        self.network = nx.Graph(edges)
        self.degrees = [4, 2, 4, 2, 2]

    def test_default(self):
        shares = cvs.OpinionSharesByDegree(3, self.network)
        x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
        c = np.array([[0, 2, 1, 2, 0, 0], [2, 1, 0, 0, 2, 0]])
        self.assertTrue(np.allclose(shares(x), c))

    def test_normalize(self):
        shares = cvs.OpinionSharesByDegree(3, self.network, normalize=True)
        x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
        c = np.array([[0, 2 / 3, 1 / 3, 1, 0, 0], [2 / 3, 1 / 3, 0, 0, 1, 0]])
        self.assertTrue(np.allclose(shares(x), c))

    def test_idx_to_return(self):
        shares = cvs.OpinionSharesByDegree(3, self.network, idx_to_return=0)
        x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
        c = np.array([[0, 2], [2, 0]])
        self.assertTrue(np.allclose(shares(x), c))

        shares = cvs.OpinionSharesByDegree(
            3, self.network, idx_to_return=np.array([2, 0])
        )
        x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
        c = np.array([[1, 0, 0, 2], [0, 2, 0, 0]])
        self.assertTrue(np.allclose(shares(x), c))


class TestCompositeCollectiveVariable(TestCase):
    def setUp(self):
        weights1 = np.array([0, 1, 0, 1, 1])
        self.shares1 = cvs.OpinionShares(
            num_opinions=2, weights=weights1, idx_to_return=0
        )
        weights2 = np.array([1, 1, 0, 0, 0])
        self.shares2 = cvs.OpinionShares(
            num_opinions=2, weights=weights2, idx_to_return=0
        )

    def test_composite_cv(self):
        composite = cvs.CompositeCollectiveVariable([self.shares1, self.shares2])
        x = np.array([[0, 1, 0, 0, 1], [1, 1, 0, 0, 0]])
        c = np.array([[1, 1], [2, 0]])
        self.assertTrue(np.allclose(composite(x), c))


class TestInterfaces(TestCase):
    def setUp(self):
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4)]
        self.network = nx.Graph(edges)
        self.degrees = [4, 2, 4, 2, 2]

    def test_default(self):
        interfaces = cvs.Interfaces(self.network)
        x = np.array([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
        c = np.array([[6], [4]])
        self.assertTrue(np.allclose(interfaces(x), c))

        with self.assertRaises(ValueError):
            x = np.array([[0, 1, 0, 1, 2]])
            interfaces(x)

    def test_normalize(self):
        interfaces = cvs.Interfaces(self.network, normalize=True)
        x = np.array([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
        c = np.array([[6 / 7], [4 / 7]])
        self.assertTrue(np.allclose(interfaces(x), c))


class TestPropensities(TestCase):
    def setUp(self):
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4)]
        self.network = nx.Graph(edges)
        self.degrees = [4, 2, 4, 2, 2]

        self.r = 1
        self.r_tilde = 0.1

    def test_complete_network(self):
        params = CNVMParameters(
            num_opinions=2, num_agents=5, r=self.r, r_tilde=self.r_tilde
        )
        propensities = cvs.Propensities(params)
        x = np.array([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        c = np.array([[1.1, 1.4], [1.7, 1.8]])
        self.assertTrue(np.allclose(propensities(x), c))

    def test_network(self):
        params = CNVMParameters(
            num_opinions=2,
            network=self.network,
            r=self.r,
            r_tilde=self.r_tilde,
            alpha=0,
        )
        propensities = cvs.Propensities(params)
        x = np.array([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
        c = np.array([[6.2, 6.3], [4.2, 4.3]])
        self.assertTrue(np.allclose(propensities(x), c))

    def test_network_normalized(self):
        params = CNVMParameters(
            num_opinions=2,
            network=self.network,
            r=self.r,
            r_tilde=self.r_tilde,
            alpha=0,
        )
        propensities = cvs.Propensities(params, True)
        x = np.array([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
        c = np.array([[6.2, 6.3], [4.2, 4.3]]) / 5.0
        self.assertTrue(np.allclose(propensities(x), c))
