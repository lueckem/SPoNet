import pytest

import sponet.collective_variables as cvs

from .utils import assert_close


@pytest.mark.parametrize(
    "x,expected",
    [
        ([[0, 1, 0, 0, 1], [1, 1, 0, 0, 0]], [[1, 1], [2, 0]]),
        ([0, 1, 0, 0, 1], [1, 1]),
    ],
)
def test_composite_cv(x, expected):
    shares1 = cvs.OpinionShares(
        num_opinions=2, weights=[0, 1, 0, 1, 1], idx_to_return=0
    )
    shares2 = cvs.OpinionShares(
        num_opinions=2, weights=[1, 1, 0, 0, 0], idx_to_return=0
    )
    cv = cvs.CompositeCollectiveVariable([shares1, shares2])
    assert_close(x, cv, expected)
