from unittest import TestCase
import numpy as np
from sponet import CNVMParameters, sample_cle, sample_stochastic_approximation


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

        t, c = sample_stochastic_approximation(params, initial_state, t_max)
        print(t)
