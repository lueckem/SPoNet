from unittest import TestCase

import networkx as nx
import numpy as np
import pytest

import sponet.network_generator as ng
from sponet.cnvm.parameters import (
    CNVMParameters,
    convert_rate_from_cnvm,
    convert_rate_to_cnvm,
)


def test_network_complete():
    params = CNVMParameters(3, num_agents=100, r=1, r_tilde=1)
    assert params.network is None
    assert params.network_generator is None
    network = params.get_network()
    assert network.number_of_edges() == 100 * (100 - 1) / 2


def test_network_network():
    network = nx.star_graph(99)
    params = CNVMParameters(3, num_agents=10, network=network, r=1, r_tilde=1)
    assert params.num_agents == 100  # used the actual network
    assert params.network_generator is None
    assert nx.utils.graphs_equal(network, params.get_network())


def test_network_generator():
    network = nx.star_graph(99)
    net_gen = ng.ErdosRenyiGenerator(100, 0.1)
    params = CNVMParameters(
        3, network=network, network_generator=net_gen, r=1, r_tilde=1
    )
    assert params.num_agents == 100
    assert params.network is None  # generator takes precedence
    net = params.get_network()
    assert net.number_of_nodes() == 100


def test_error_no_network():
    with pytest.raises(ValueError):
        CNVMParameters(3, r=1, r_tilde=1)


@pytest.mark.parametrize(
    "in_num_opinions,in_r,in_r_tilde,r,r_tilde",
    [
        (
            3,
            1,
            0.1,
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]],
        ),
        (
            3,
            [[1, 1, 1], [1, 2, 3], [4, 5, 6]],
            [[0, 0.1, 0.2], [0.3, -2, 0.4], [0.5, 0.6, 0]],
            [[0, 1, 1], [1, 0, 3], [4, 5, 0]],
            [[0, 0.1, 0.2], [0.3, 0, 0.4], [0.5, 0.6, 0]],
        ),
        (
            None,
            [[1, 1], [2, 3]],
            [[0.1, 0], [0.3, -2]],
            [[0, 1], [2, 0]],
            [[0, 0], [0.3, 0]],
        ),
    ],
)
def test_rates_style1(
    in_num_opinions,
    in_r,
    in_r_tilde,
    r,
    r_tilde,
):
    params = CNVMParameters(in_num_opinions, r=in_r, r_tilde=in_r_tilde, num_agents=10)
    assert np.all(params.r == r)
    assert np.all(params.r_tilde == r_tilde)
    assert params.num_opinions == np.array(r).shape[0]

    (
        r_imit,
        r_noise,
        prob_imit,
        prob_noise,
    ) = convert_rate_to_cnvm(r, r_tilde)
    assert params.r_imit == r_imit
    assert params.r_noise == r_noise
    assert np.all(params.prob_imit == prob_imit)
    assert np.all(params.prob_noise == prob_noise)


@pytest.mark.parametrize(
    "num_opinions,r,r_tilde",
    [
        (None, 1, 0.1),
        (2, -1, 0.1),
        (2, 1, -0.1),
        (3, [[0, 1], [2, 0]], [[0, 0], [0.3, 0]]),
        (None, [[0, -1], [2, 0]], [[0, 0], [0.3, 0]]),
        (None, [0, 1], [[0, 0], [0.3, 0]]),
        (None, [[0, 1], [0, 1], [0, 1]], [[0, 0], [0.3, 0]]),
    ],
)
def test_errors_rates_style1(num_opinions, r, r_tilde):
    with pytest.raises(ValueError):
        CNVMParameters(num_opinions, 100, r=r, r_tilde=r_tilde)


@pytest.mark.parametrize(
    "in_num_opinions,in_prob_imit,in_prob_noise,prob_imit,prob_noise",
    [
        (
            3,
            1,
            0.1,
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]],
        ),
        (
            3,
            [[1, 1, 1], [1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[1, 1, 1], [1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0, 1, 1], [1, 0, 0.3], [0.4, 0.5, 0]],
            [[0, 1, 1], [1, 0, 0.3], [0.4, 0.5, 0]],
        ),
        (
            None,
            [[1, 1], [0.2, 0.3]],
            [[0.1, 0], [0.3, -2]],
            [[0, 1], [0.2, 0]],
            [[0, 0], [0.3, 0]],
        ),
    ],
)
def test_rates_style2(
    in_num_opinions,
    in_prob_imit,
    in_prob_noise,
    prob_imit,
    prob_noise,
):
    r_imit = 1
    r_noise = 0.1
    params = CNVMParameters(
        in_num_opinions,
        r_imit=r_imit,
        r_noise=r_noise,
        prob_imit=in_prob_imit,
        prob_noise=in_prob_noise,
        num_agents=10,
    )
    assert params.r_imit == r_imit
    assert params.r_noise == r_noise
    assert np.all(params.prob_noise == prob_noise)
    assert np.all(params.prob_imit == prob_imit)
    num_opinions = np.array(prob_imit).shape[0]
    assert params.num_opinions == num_opinions

    r = r_imit * np.array(prob_imit)
    r_tilde = r_noise * np.array(prob_noise) / num_opinions
    assert np.all(params.r == r)
    assert np.all(params.r_tilde == r_tilde)


@pytest.mark.parametrize(
    "num_opinions,prob_imit,prob_noise",
    [
        (None, 0.2, 0.1),
        (2, 2, 2),
        (2, -1, 0.1),
        (2, 1, -0.1),
        (3, [[0, 1], [0.2, 0]], [[0, 0], [0.3, 0]]),
        (None, [[0, -1], [0.2, 0]], [[0, 0], [0.3, 0]]),
        (None, [[0, 1], [2, 0]], [[0, 0], [0.3, 0]]),
        (None, [[0, 1], [0.2, 0]], [[0, 0], [-0.3, 0]]),
        (None, [0, 1], [[0, 0], [0.3, 0]]),
        (None, [[0, 1], [0, 1], [0, 1]], [[0, 0], [0.3, 0]]),
    ],
)
def test_errors_rates_style2(num_opinions, prob_imit, prob_noise):
    with pytest.raises(ValueError):
        CNVMParameters(
            num_opinions,
            100,
            r_imit=1,
            r_noise=0.1,
            prob_imit=prob_imit,
            prob_noise=prob_noise,
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
