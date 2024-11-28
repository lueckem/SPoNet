from unittest import TestCase

import numpy as np

import sponet.steady_state as st
from sponet.cnvm.parameters import CNVMParameters
from sponet.collective_variables import OpinionShares


class TestSteadyState(TestCase):
    def test_1(self):
        params = CNVMParameters(
            num_opinions=2,
            num_agents=10000,
            r=np.array([[0, 1.01], [0.99, 0]]),
            r_tilde=0.01,
        )
        col_var = OpinionShares(2, True)

        c_steady = st.estimate_steady_state(params, col_var, tol_abs=0.001)
        self.assertEqual(c_steady.shape, (2,))
