import numpy as np
from numpy.typing import ArrayLike

import sponet.collective_variables as cvs


def assert_close(x: ArrayLike, cv: cvs.CollectiveVariable, expected: ArrayLike):
    """
    Assert that cv(x) == expected.
    """
    c = cv(np.array(x))
    expected = np.array(expected)
    assert c.shape == expected.shape
    assert np.allclose(c, expected)
