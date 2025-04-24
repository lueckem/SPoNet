from unittest import TestCase

import numpy as np

from sponet import CNVMParameters
from sponet.collective_variables import OpinionShares
from sponet.sample_moments import sample_moments


class TestSampleMoments(TestCase):
    def setUp(self):
        self.num_opinions = 3
        self.num_agents = 100
        self.r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
        self.r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

        self.params = CNVMParameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=self.r,
            r_tilde=self.r_tilde,
        )

        self.t_max = 100
        self.num_timesteps = 10
        self.cv = OpinionShares(self.num_opinions)
        self.initial_state = np.random.randint(3, size=(self.num_agents,))

    def test_runs(self):
        t, mean, variance, num_samples = sample_moments(
            params=self.params,
            initial_state=self.initial_state,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            batch_size=100,
            rel_tol=0.1,
            n_jobs=1,
            collective_variable=self.cv,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(mean.shape, (self.num_timesteps, self.num_opinions))
        self.assertEqual(variance.shape, (self.num_timesteps, self.num_opinions))
