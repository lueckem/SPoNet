from unittest import TestCase
import numpy as np
import networkx as nx

import sponet.collective_variables as cv
from sponet.cnvm.parameters import CNVMParameters


class TestOpinionShares(TestCase):
    def test_default(self):
        shares = cv.OpinionShares(3)

        x = np.array([[0, 1, 0, 1, 2, 2, 1, 1, 1, 0]])
        self.assertTrue(np.allclose(shares(x), np.array([[3, 5, 2]])))

        x = np.array([[0, 1, 0, 1, 0, 0, 0], [1, 1, 1, 2, 1, 2, 2]])
        c = np.array([[5, 2, 0], [0, 4, 3]])
        self.assertTrue(np.allclose(shares(x), c))

        x = np.array([[1, 1, 1, 2, 1, 2, 2]])
        self.assertTrue(np.allclose(shares(x), np.array([[0, 4, 3]])))

        x = np.array([[]])
        self.assertTrue(np.allclose(shares(x), np.array([[0, 0, 0]])))

    def test_normalized(self):
        shares = cv.OpinionShares(3, normalize=True)

        x = np.array([[0, 1, 0, 1, 2, 2, 1, 1, 1, 0]])
        self.assertTrue(np.allclose(shares(x), np.array([[0.3, 0.5, 0.2]])))

        x = np.array([[0, 1, 0, 1, 0]])
        self.assertTrue(np.allclose(shares(x), np.array([[0.6, 0.4, 0]])))

    def test_weights(self):
        weights = np.array([1, 0, 0.5, 0, 1])
        shares = cv.OpinionShares(3, weights=weights)

        x = np.array([[0, 1, 0, 2, 1]])
        self.assertTrue(np.allclose(shares(x), np.array([[1.5, 1, 0]])))

        shares = cv.OpinionShares(3, weights=weights, normalize=True)
        x = np.array([[0, 0, 1, 1, 2]])
        self.assertTrue(
            np.allclose(shares(x), np.array([[1 / 2.5, 0.5 / 2.5, 1 / 2.5]]))
        )

    def test_idx_to_return(self):
        x = np.array([[0, 1, 0, 1, 2, 2, 1, 1, 1, 0]])

        shares = cv.OpinionShares(3, idx_to_return=1)
        self.assertTrue(np.allclose(shares(x), np.array([[5]])))

        shares = cv.OpinionShares(3, idx_to_return=np.array([0, 1]))
        self.assertTrue(np.allclose(shares(x), np.array([[3, 5]])))

        shares = cv.OpinionShares(3, idx_to_return=np.array([2, 0, 1]))
        self.assertTrue(np.allclose(shares(x), np.array([[2, 3, 5]])))


class TestOpinionSharesByDegree(TestCase):
    def setUp(self):
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4)]
        self.network = nx.Graph(edges)
        self.degrees = [4, 2, 4, 2, 2]

    def test_default(self):
        shares = cv.OpinionSharesByDegree(3, self.network)
        x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
        c = np.array([[0, 2, 1, 2, 0, 0], [2, 1, 0, 0, 2, 0]])
        self.assertTrue(np.allclose(shares(x), c))

    def test_normalize(self):
        shares = cv.OpinionSharesByDegree(3, self.network, normalize=True)
        x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
        c = np.array([[0, 2 / 3, 1 / 3, 1, 0, 0], [2 / 3, 1 / 3, 0, 0, 1, 0]])
        self.assertTrue(np.allclose(shares(x), c))

    def test_idx_to_return(self):
        shares = cv.OpinionSharesByDegree(3, self.network, idx_to_return=0)
        x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
        c = np.array([[0, 2], [2, 0]])
        self.assertTrue(np.allclose(shares(x), c))

        shares = cv.OpinionSharesByDegree(
            3, self.network, idx_to_return=np.array([2, 0])
        )
        x = np.array([[0, 1, 0, 2, 1], [1, 1, 1, 0, 0]])
        c = np.array([[1, 0, 0, 2], [0, 2, 0, 0]])
        self.assertTrue(np.allclose(shares(x), c))


class TestCompositeCollectiveVariable(TestCase):
    def setUp(self):
        weights1 = np.array([0, 1, 0, 1, 1])
        self.shares1 = cv.OpinionShares(
            num_opinions=2, weights=weights1, idx_to_return=0
        )
        weights2 = np.array([1, 1, 0, 0, 0])
        self.shares2 = cv.OpinionShares(
            num_opinions=2, weights=weights2, idx_to_return=0
        )

    def test_composite_cv(self):
        composite = cv.CompositeCollectiveVariable([self.shares1, self.shares2])
        x = np.array([[0, 1, 0, 0, 1], [1, 1, 0, 0, 0]])
        c = np.array([[1, 1], [2, 0]])
        self.assertTrue(np.allclose(composite(x), c))


class TestInterfaces(TestCase):
    def setUp(self):
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4)]
        self.network = nx.Graph(edges)
        self.degrees = [4, 2, 4, 2, 2]

    def test_default(self):
        interfaces = cv.Interfaces(self.network)
        x = np.array([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
        c = np.array([[6], [4]])
        self.assertTrue(np.allclose(interfaces(x), c))

        with self.assertRaises(ValueError):
            x = np.array([[0, 1, 0, 1, 2]])
            interfaces(x)

    def test_normalize(self):
        interfaces = cv.Interfaces(self.network, normalize=True)
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
        propensities = cv.Propensities(params)
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
        propensities = cv.Propensities(params)
        x = np.array([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
        c = np.array([[6.2, 6.3], [4.2, 4.3]])
        self.assertTrue(np.allclose(propensities(x), c))
