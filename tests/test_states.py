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
        x = ss.sample_states_local_clusters(network, 3, 2)
        self.assert_states_valid(x, num_agents, 3, 2)
