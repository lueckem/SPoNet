from unittest import TestCase
import numpy as np
import networkx as nx

import sponet.states as ss


class TestStateSampling(TestCase):
    def assert_states_valid(
        self, states: np.ndarray, num_agents: int, num_opinions: int, num_states: int
    ):
        self.assertEqual(states.shape, (num_states, num_agents))
        self.assertTrue(np.issubdtype(states.dtype, np.integer))
        self.assertTrue(np.all(states >= 0))
        self.assertTrue(np.all(states < num_opinions))

    def test_sample_states_uniform(self):
        x = ss.sample_states_uniform(100, 3, 2)
        self.assert_states_valid(x, 100, 3, 2)

    def test_sample_states_uniform_shares(self):
        x = ss.sample_states_uniform_shares(100, 3, 2)
        self.assert_states_valid(x, 100, 3, 2)

    def test_sample_states_local_cluster(self):
        num_agents = 100
        network = nx.watts_strogatz_graph(num_agents, 4, 0)
        x = ss.sample_states_local_clusters(
            network, 3, 2, min_num_seeds=2, max_num_seeds=5
        )
        self.assert_states_valid(x, num_agents, 3, 2)

    def test_build_state_by_degree_valid_shape(self):
        num_agents = 100
        opinion_shares = np.array([0.2, 0.3, 0.5])
        opinion_order = np.array([1, 0, 2])
        network = nx.barabasi_albert_graph(num_agents, 3)
        x = ss.build_state_by_degree(network, opinion_shares, opinion_order)
        self.assertEqual(x.shape, (num_agents,))
        self.assertTrue(np.issubdtype(x.dtype, np.integer))
        self.assertTrue(np.all(x >= 0))
        self.assertTrue(np.all(x < 3))

    def test_build_state_by_degree_example(self):
        network = nx.Graph()
        network.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
            ]
        )  # degrees = [4, 3, 2, 2, 1]

        opinion_shares = np.array([0.4, 0.4, 0.2])
        opinion_order = np.array([1, 0, 2])
        x = ss.build_state_by_degree(network, opinion_shares, opinion_order)
        self.assertTrue(np.all(x == np.array([1, 1, 0, 0, 2])))
