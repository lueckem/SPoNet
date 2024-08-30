from unittest import TestCase
import numpy as np

from sponet import CNVMParameters, sample_many_runs
from sponet.collective_variables import OpinionShares
import sponet.multiprocessing as smp


class TestSampleManyRuns(TestCase):
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
        self.num_initial_states = 5
        self.initial_states = np.random.randint(
            2,
            size=(self.num_initial_states, self.num_agents),
        )

    def test_split_runs(self):
        num_runs, num_chunks = 20, 6
        chunks = smp._split_runs(num_runs, num_chunks)
        self.assertTrue(np.allclose(chunks, [4, 4, 3, 3, 3, 3]))

    def test_parallelization_runs(self):
        t, x = sample_many_runs(
            params=self.params,
            initial_states=self.initial_states,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            num_runs=15,
            n_jobs=2,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(
            x.shape,
            (
                self.num_initial_states,
                15,
                self.num_timesteps,
                self.num_agents,
            ),
        )

    def test_parallelization_initial_states(self):
        t, x = sample_many_runs(
            params=self.params,
            initial_states=self.initial_states,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            num_runs=3,
            n_jobs=2,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(
            x.shape,
            (
                self.num_initial_states,
                3,
                self.num_timesteps,
                self.num_agents,
            ),
        )

    def test_no_parallelization(self):
        t, x = sample_many_runs(
            params=self.params,
            initial_states=self.initial_states,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            num_runs=3,
            n_jobs=None,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(
            x.shape,
            (
                self.num_initial_states,
                3,
                self.num_timesteps,
                self.num_agents,
            ),
        )

    def test_parallelization_with_cv(self):
        t, x = sample_many_runs(
            params=self.params,
            initial_states=self.initial_states,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            num_runs=15,
            n_jobs=2,
            collective_variable=self.cv,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(
            x.shape,
            (
                self.num_initial_states,
                15,
                self.num_timesteps,
                self.num_opinions,
            ),
        )

    def test_output_dtype_no_cv(self):
        num_opinions_list = [2, 257]
        correct_dtype_list = [np.uint8, np.uint16]

        for num_opinions, correct_dtype in zip(num_opinions_list, correct_dtype_list):

            params = CNVMParameters(
                num_opinions=num_opinions,
                num_agents=self.num_agents,
                r=1,
                r_tilde=1,
            )

            t, x = sample_many_runs(
                params=params,
                initial_states=self.initial_states,
                t_max=5,
                num_timesteps=self.num_timesteps,
                num_runs=2,
                n_jobs=2,
                collective_variable=None,
            )

            self.assertEqual(correct_dtype, x.dtype)

