from unittest import TestCase
import numpy as np
from sponet.utils import argmatch


class TestArgmatch(TestCase):
    def test_1(self):
        t = np.array([1, 2, 3, 4, 5, 6, 7])
        t_ref = np.array([1.8, 2, 4.4, 6.7, 6.9])
        ind = argmatch(t_ref, t)
        self.assertTrue(np.allclose(ind, [1, 1, 3, 6, 6]))

    def test_2(self):
        t = np.array([1, 2, 3, 4, 5])
        t_ref = np.array([0.1, 1.8, 2, 2.5, 4.4, 6.7, 7.7])
        ind = argmatch(t_ref, t)
        self.assertTrue(np.allclose(ind, [0, 1, 1, 2, 3, 4, 4]))
