from unittest import TestCase
import numpy as np
import networkx as nx

import sponet.states as ss


class TestStateSampling(TestCase):
    def test_sample_states_uniform(self):
        x = ss.sample_states_uniform(100, 3, 2)
        self.assertEqual(x.shape, (2, 100))
        self.assertTrue(np.issubdtype(x.dtype, np.integer))
        self.assertTrue(np.all(x >= 0))
        self.assertTrue(np.all(x < 3))
