from unittest import TestCase

import networkx as nx
import numpy as np

from sponet import (
    CNVMParameters,
    calc_pair_approximation_traj,
    calc_rre_traj,
    sample_cle,
    sample_many_runs,
    sample_stochastic_approximation,
)
from sponet.collective_variables import OpinionShares


class TestCLE(TestCase):
    def test_sample_cle(self):
        num_opinions = 3
        num_agents = 100
        r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
        r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

        params = CNVMParameters(
            num_opinions=num_opinions,
            num_agents=num_agents,
            r=r,
            r_tilde=r_tilde,
        )

        t_max = 100
        num_time_steps = 10000
        initial_state = np.array([0.9, 0.1, 0.0])
        num_samples = 5

        t, c = sample_cle(params, initial_state, t_max, num_time_steps, num_samples)
        self.assertTrue(np.allclose(t, np.linspace(0, t_max, num_time_steps + 1)))
        self.assertEqual(c.shape, (num_samples, num_time_steps + 1, num_opinions))
        self.assertTrue(np.allclose(c[0, 0, :], initial_state))


class TestStochasticApprox(TestCase):
    def test_sample(self):
        num_opinions = 3
        num_agents = 100
        r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
        r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

        params = CNVMParameters(
            num_opinions=num_opinions,
            num_agents=num_agents,
            r=r,
            r_tilde=r_tilde,
        )

        t_max = 100
        initial_state = np.array([0.9, 0.1, 0.0])
        num_time_steps = 51
        num_samples = 25

        t, c = sample_stochastic_approximation(
            params, initial_state, t_max, num_time_steps, num_samples
        )
        self.assertTrue(np.allclose(t, np.linspace(0, t_max, num_time_steps + 1)))
        self.assertEqual(c.shape, (num_samples, num_time_steps + 1, num_opinions))
        self.assertTrue(np.allclose(c[0, 0, :], initial_state))


class TestAgreement(TestCase):
    def test_agreement_between_approximations(self):
        num_opinions = 3
        num_agents = 100000
        r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
        r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

        params = CNVMParameters(
            num_opinions=num_opinions,
            num_agents=num_agents,
            r=r,
            r_tilde=r_tilde,
        )
        cv = OpinionShares(num_opinions, True)
        t_max = 30
        num_time_steps = 30
        num_samples = 100

        initial_c = np.array([0.9, 0.1, 0.0])
        num_0 = int(initial_c[0] * num_agents)
        num_1 = int(initial_c[1] * num_agents)
        num_2 = num_agents - num_0 - num_1
        initial_x = np.array([0] * num_0 + [1] * num_1 + [2] * num_2)

        t, c = sample_many_runs(
            params,
            np.array([initial_x]),
            t_max,
            num_time_steps + 1,
            num_samples,
            n_jobs=-1,
            collective_variable=cv,
        )
        t_cle, c_cle = sample_cle(
            params,
            initial_c,
            t_max,
            num_time_steps * 100,
            num_samples,
            saving_offset=100,
        )
        t_sa, c_sa = sample_stochastic_approximation(
            params, initial_c, t_max, num_time_steps, num_samples
        )
        t_rre, c_rre = calc_rre_traj(
            params, initial_c, t_max, np.linspace(0, t_max, num_time_steps + 1)
        )

        mean_c = np.mean(c[0], axis=0)
        mean_c_cle = np.mean(c_cle, axis=0)
        mean_c_sa = np.mean(c_sa, axis=0)

        self.assertTrue(np.allclose(t, t_cle))
        self.assertTrue(np.allclose(t, t_sa))
        self.assertTrue(np.allclose(t, t_rre))

        self.assertTrue(np.mean((mean_c - mean_c_cle) ** 2) < 1e-3)
        self.assertTrue(np.mean((mean_c - mean_c_sa) ** 2) < 1e-3)
        self.assertTrue(np.mean((mean_c - c_rre) ** 2) < 1e-3)


class TestPairApprox(TestCase):
    def test_calc_pair_approximation_traj(self):
        from sponet.collective_variables import Interfaces

        num_opinions = 2
        num_agents = 1000
        r = np.array([[0, 1], [1.1, 0]])
        r_tilde = 0.01
        network = nx.barabasi_albert_graph(num_agents, 3)

        params = CNVMParameters(
            num_opinions=num_opinions,
            network=network,
            r=r,
            r_tilde=r_tilde,
        )

        t_max = 100
        num_time_steps = 10000
        x_0 = np.zeros(num_agents)
        x_0[:100] = 1
        c_0 = 0.1
        s_0 = 0.5 * Interfaces(network, True)(np.array([x_0]))[0, 0]
        mean_degree = np.mean([d for _, d in network.degree()])

        t, c_pa = calc_pair_approximation_traj(
            params,
            c_0,
            s_0,
            mean_degree,
            t_max,
            t_eval=np.linspace(0, t_max, num_time_steps + 1),
        )

        self.assertEqual(c_pa.shape, (10001, 2))
        self.assertEqual(c_pa[0, 0], c_0)
        self.assertEqual(c_pa[0, 1], s_0)
        self.assertTrue(np.allclose(t, np.linspace(0, t_max, num_time_steps + 1)))
