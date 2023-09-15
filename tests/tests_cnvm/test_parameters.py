from unittest import TestCase
import numpy as np
import networkx as nx

from sponet.cnvm.parameters import (
    CNVMParameters,
    convert_rate_to_cnvm,
    convert_rate_from_cnvm,
)
import sponet.network_generator as ng


class TestParametersNetworks(TestCase):
    def setUp(self):
        self.num_opinions = 3
        self.num_agents = 100
        self.r_imit = 1
        self.r_noise = 0.1

    def test_complete(self):
        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r_imit=self.r_imit,
            r_noise=self.r_noise,
        )

        self.assertIsNone(params.network)
        self.assertIsNone(params.network_generator)

        network = params.get_network()
        self.assertEqual(
            network.number_of_edges(), self.num_agents * (self.num_agents - 1) / 2
        )

    def test_network(self):
        network = nx.star_graph(self.num_agents - 1)

        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=3,  # should be overwritten by correct number
            network=network,
            r_imit=self.r_imit,
            r_noise=self.r_noise,
        )

        self.assertEqual(params.num_agents, self.num_agents)
        self.assertIsNone(params.network_generator)
        self.assertTrue(nx.utils.graphs_equal(network, params.get_network()))

    def test_graph_generator(self):
        network = nx.star_graph(self.num_agents - 1)
        net_gen = ng.ErdosRenyiGenerator(self.num_agents, 0.1)

        params = CNVMParameters(
            num_opinions=self.num_opinions,
            network=network,
            network_generator=net_gen,  # net_gen should overwrite network
            r_imit=self.r_imit,
            r_noise=self.r_noise,
        )

        self.assertEqual(params.num_agents, self.num_agents)
        self.assertIsNone(params.network)
        network = params.get_network()
        self.assertTrue(network.number_of_nodes(), self.num_agents)

    def test_no_network(self):
        with self.assertRaises(ValueError):
            CNVMParameters(
                num_opinions=self.num_opinions, r_imit=self.r_imit, r_noise=self.r_noise
            )


class TestParametersRates(TestCase):
    def setUp(self):
        self.num_opinions = 2
        self.num_agents = 2

    def test_rates_style1_float(self):
        r = 1
        r_tilde = 0.1

        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=r,
            r_tilde=r_tilde,
        )
        r_array = np.array([[0, 1], [1, 0]])
        r_tilde_array = np.array([[0, 0.1], [0.1, 0]])
        self.assertTrue(np.allclose(params.r, r_array))
        self.assertTrue(np.allclose(params.r_tilde, r_tilde_array))

        r_imit = 1
        r_noise = 0.1 * self.num_opinions
        prob_ones = np.array([[0, 1], [1, 0]])
        self.assertTrue(np.allclose(params.r_imit, r_imit))
        self.assertTrue(np.allclose(params.r_noise, r_noise))
        self.assertTrue(np.allclose(params.prob_imit, prob_ones))
        self.assertTrue(np.allclose(params.prob_noise, prob_ones))

    def test_rates_style1_array(self):
        r = np.array([[0, 2], [4, 1000]])  # diagonal should be ignored
        r_tilde = np.array([[1000, 0.2], [0.1, 0]])

        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=r,
            r_tilde=r_tilde,
        )
        r_array = np.array([[0, 2], [4, 0]])
        r_tilde_array = np.array([[0, 0.2], [0.1, 0]])
        self.assertTrue(np.allclose(params.r, r_array))
        self.assertTrue(np.allclose(params.r_tilde, r_tilde_array))

        r_imit = 4
        r_noise = 0.2 * self.num_opinions
        prob_imit = np.array([[0, 0.5], [1, 0]])
        prob_noise = np.array([[0, 1], [0.5, 0]])
        self.assertTrue(np.allclose(params.r_imit, r_imit))
        self.assertTrue(np.allclose(params.r_noise, r_noise))
        self.assertTrue(np.allclose(params.prob_imit, prob_imit))
        self.assertTrue(np.allclose(params.prob_noise, prob_noise))

    def test_style1_exceptions(self):
        r = np.array([[0, 2], [4, 0]])
        r_tilde = np.array([[0, -0.2], [0.1, 0]])

        with self.assertRaises(ValueError):
            CNVMParameters(
                num_opinions=self.num_opinions,
                num_agents=self.num_agents,
                r=r,
                r_tilde=r_tilde,
            )

    def test_rates_style2_float(self):
        r_imit = 1
        r_noise = 0.2

        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r_imit=r_imit,
            r_noise=r_noise,
        )

        prob_ones = np.array([[0, 1], [1, 0]])
        self.assertTrue(np.allclose(params.r_imit, r_imit))
        self.assertTrue(np.allclose(params.r_noise, r_noise))
        self.assertTrue(np.allclose(params.prob_imit, prob_ones))
        self.assertTrue(np.allclose(params.prob_noise, prob_ones))

        r_array = np.array([[0, 1], [1, 0]])
        r_tilde_array = np.array([[0, 0.1], [0.1, 0]])
        self.assertTrue(np.allclose(params.r, r_array))
        self.assertTrue(np.allclose(params.r_tilde, r_tilde_array))

    def test_rates_style2_array(self):
        r_imit = 1
        r_noise = 0.2
        prob_imit = np.array([[0, 0.2], [1, 1000]])  # diagonal should be ignored
        prob_noise = np.array([[-10, 0.9], [1, 0]])

        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r_imit=r_imit,
            r_noise=r_noise,
            prob_imit=prob_imit,
            prob_noise=prob_noise,
        )

        prob_imit = np.array([[0, 0.2], [1, 0]])
        prob_noise = np.array([[0, 0.9], [1, 0]])
        self.assertTrue(np.allclose(params.r_imit, r_imit))
        self.assertTrue(np.allclose(params.r_noise, r_noise))
        self.assertTrue(np.allclose(params.prob_imit, prob_imit))
        self.assertTrue(np.allclose(params.prob_noise, prob_noise))

        r_array = np.array([[0, 0.2], [1, 0]])
        r_tilde_array = np.array([[0, 0.09], [0.1, 0]])
        self.assertTrue(np.allclose(params.r, r_array))
        self.assertTrue(np.allclose(params.r_tilde, r_tilde_array))

    def test_style2_exceptions(self):
        with self.assertRaises(ValueError):
            CNVMParameters(
                num_opinions=self.num_opinions,
                num_agents=self.num_agents,
                r_imit=-0.2,
                r_noise=0.1,
            )

        with self.assertRaises(ValueError):
            CNVMParameters(
                num_opinions=self.num_opinions,
                num_agents=self.num_agents,
                r_imit=1,
                r_noise=-0.1,
            )

        prob_imit = np.array([[0, 0.2], [1.5, 0]])
        with self.assertRaises(ValueError):
            CNVMParameters(
                num_opinions=self.num_opinions,
                num_agents=self.num_agents,
                r_imit=1,
                r_noise=0.1,
                prob_imit=prob_imit,
            )

        prob_noise = np.array([[0, 0.2], [1.5, 0]])
        with self.assertRaises(ValueError):
            CNVMParameters(
                num_opinions=self.num_opinions,
                num_agents=self.num_agents,
                r_imit=1,
                r_noise=0.1,
                prob_noise=prob_noise,
            )

        with self.assertRaises(ValueError):
            CNVMParameters(
                num_opinions=self.num_opinions,
                num_agents=self.num_agents,
                r_imit=1,
                r=1,
            )


class TestParametersGetSet(TestCase):
    def setUp(self):
        self.num_opinions = 3
        self.num_agents = 100
        self.r = 1
        self.r_tilde = 0.1

    def test_change_rates(self):
        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=self.r,
            r_tilde=self.r_tilde,
        )
        new_r = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]])
        new_r_tilde = 0.2

        params.change_rates(r=new_r, r_tilde=new_r_tilde)
        self.assertTrue(np.allclose(params.r, new_r))
        new_r_tilde = np.array([[0, 0.2, 0.2], [0.2, 0, 0.2], [0.2, 0.2, 0]])
        self.assertTrue(np.allclose(params.r_tilde, new_r_tilde))

    def test_get_network(self):
        # complete network
        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=self.r,
            r_tilde=self.r_tilde,
        )
        network = params.get_network()
        self.assertEqual(
            network.number_of_edges(), self.num_agents * (self.num_agents - 1) / 2
        )

        # network
        network = nx.star_graph(self.num_agents - 1)
        params = CNVMParameters(
            num_opinions=self.num_opinions,
            network=network,
            r=self.r,
            r_tilde=self.r_tilde,
        )
        self.assertTrue(nx.utils.graphs_equal(network, params.get_network()))

        # network generator
        params = CNVMParameters(
            num_opinions=self.num_opinions,
            network_generator=ng.ErdosRenyiGenerator(self.num_agents, 0.1),
            r=self.r,
            r_tilde=self.r_tilde,
        )
        network = params.get_network()
        self.assertEqual(network.number_of_nodes(), self.num_agents)

    def test_set_network(self):
        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=self.r,
            r_tilde=self.r_tilde,
        )
        network = nx.star_graph(self.num_agents - 1)
        params.set_network(network)
        self.assertTrue(nx.utils.graphs_equal(network, params.get_network()))

        params = CNVMParameters(
            num_opinions=self.num_opinions,
            network_generator=ng.ErdosRenyiGenerator(self.num_agents, 0.1),
            r=self.r,
            r_tilde=self.r_tilde,
        )
        with self.assertRaises(ValueError):
            params.set_network(network)

    def test_update_network(self):
        params = CNVMParameters(
            num_opinions=self.num_opinions,
            network_generator=ng.ErdosRenyiGenerator(self.num_agents, 0.1),
            r=self.r,
            r_tilde=self.r_tilde,
        )
        params.update_network_by_generator()
        self.assertIsNotNone(params.network)

        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=self.r,
            r_tilde=self.r_tilde,
        )
        with self.assertRaises(ValueError):
            params.update_network_by_generator()


class TestConversionOfRates(TestCase):
    def setUp(self):
        self.num_opinions = 3
        self.r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
        self.r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

        self.r_imit = 2
        self.r_noise = 0.6
        self.prob_imit = np.array([[0, 0.5, 1], [0.5, 0, 0.5], [1, 0, 0]])
        self.prob_noise = np.array([[0, 1, 0.5], [0, 0, 0.5], [0.5, 1, 0]])

    def test_style1_to_style2(self):
        this_r = np.copy(self.r)
        this_r[1, 1] = 1000  # diagonal entries should be ignored
        r_imit, r_noise, prob_imit, prob_noise = convert_rate_to_cnvm(
            this_r, self.r_tilde
        )
        self.assertTrue(np.allclose(r_imit, self.r_imit))
        self.assertTrue(np.allclose(r_noise, self.r_noise))
        self.assertTrue(np.allclose(prob_imit, self.prob_imit))
        self.assertTrue(np.allclose(prob_noise, self.prob_noise))

    def test_style2_to_style1(self):
        params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=10,
            r_imit=self.r_imit,
            r_noise=self.r_noise,
            prob_imit=self.prob_imit,
            prob_noise=self.prob_noise,
        )
        r, r_tilde = convert_rate_from_cnvm(params)
        self.assertTrue(np.allclose(r, self.r))
        self.assertTrue(np.allclose(r_tilde, self.r_tilde))

    def test_rate_0(self):
        true_r_imit = 0
        true_r_noise = 0
        prob = np.zeros((2, 2))

        r = np.zeros((2, 2))
        r_tilde = np.zeros((2, 2))

        r_imit, r_noise, prob_imit, prob_noise = convert_rate_to_cnvm(r, r_tilde)
        self.assertTrue(np.allclose(r_imit, true_r_imit))
        self.assertTrue(np.allclose(r_noise, true_r_noise))
        self.assertTrue(np.allclose(prob_imit, prob))
        self.assertTrue(np.allclose(prob_noise, prob))
