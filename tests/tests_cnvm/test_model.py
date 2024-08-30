from unittest import TestCase
import numpy as np
import networkx as nx

from sponet.cnvm.parameters import CNVMParameters
from sponet.cnvm.model import CNVM
import sponet.network_generator as ng


class TestModel(TestCase):
    def setUp(self):
        self.num_opinions = 3
        self.num_agents = 100
        self.r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
        self.r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

        self.params_complete = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=self.r,
            r_tilde=self.r_tilde,
        )

        self.params_network = CNVMParameters(
            num_opinions=self.num_opinions,
            network=nx.barabasi_albert_graph(self.num_agents, 2),
            r=self.r,
            r_tilde=self.r_tilde,
            alpha=0,
        )

        self.params_generator = CNVMParameters(
            num_opinions=self.num_opinions,
            network_generator=ng.BarabasiAlbertGenerator(self.num_agents, 2),
            r=self.r,
            r_tilde=self.r_tilde,
        )


    def test_output(self):
        model = CNVM(self.params_complete)
        t_max = 100
        t, x = model.simulate(t_max)
        self.assertEqual(t[0], 0)
        self.assertEqual(t.shape[0], x.shape[0])

        t, x = model.simulate(t_max, len_output=10)
        self.assertEqual(t.shape, (10,))
        self.assertEqual(x.shape, (10, self.num_agents))
        self.assertEqual(t[0], 0)
        self.assertTrue(t[-1] >= t_max)

        model = CNVM(self.params_network)
        t, x = model.simulate(t_max)
        self.assertEqual(t[0], 0)
        self.assertEqual(t.shape[0], x.shape[0])

        t, x = model.simulate(t_max, len_output=10)
        self.assertEqual(t.shape, (10,))
        self.assertEqual(x.shape, (10, self.num_agents))
        self.assertEqual(t[0], 0)
        self.assertTrue(t[-1] >= t_max)

        x_init = np.ones(self.num_agents)
        model = CNVM(self.params_generator)
        t, x = model.simulate(t_max, x_init=x_init)
        self.assertEqual(t[0], 0)
        self.assertEqual(t.shape[0], x.shape[0])
        self.assertTrue(np.allclose(x[0], x_init))

        t, x = model.simulate(t_max, x_init, len_output=10)
        self.assertEqual(t.shape, (10,))
        self.assertEqual(x.shape, (10, self.num_agents))
        self.assertEqual(t[0], 0)
        self.assertTrue(t[-1] >= t_max)
        self.assertTrue(np.allclose(x[0], x_init))

    def test_rng(self):
        t_max = 100

        rng1 = np.random.default_rng(1)
        model = CNVM(self.params_network)
        t1, x1 = model.simulate(t_max, rng=rng1)

        rng2 = np.random.default_rng(1)
        model = CNVM(self.params_network)
        t2, x2 = model.simulate(t_max, rng=rng2)

        self.assertTrue(np.allclose(t1, t2))
        self.assertTrue(np.allclose(x1, x2))

    def test_output_concise(self):
        # If len_output is not specified, the output should only contain states that
        # have changed from one snapshot to the next
        model = CNVM(self.params_network)
        t_max = 100
        t, x = model.simulate(t_max)

        for i in range(x.shape[0] - 1):
            self.assertFalse(np.allclose(x[i], x[i + 1]))

    def test_output_dtype(self):

        num_opinions_list = [2, 257]
        correct_dtype_list = [np.uint8, np.uint16]

        for num_opinions, correct_dtype in zip(num_opinions_list, correct_dtype_list):
            params = CNVMParameters(
                num_opinions=num_opinions,
                num_agents=self.num_agents,
                r=1,
                r_tilde=1,
            )

            model = CNVM(params)
            t_max = 5
            t, x = model.simulate(t_max)

            self.assertEqual(correct_dtype, x.dtype)








