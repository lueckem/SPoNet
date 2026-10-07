import numpy as np
import pytest

from sponet.cnvm.approximations.hybrid_jump_diffusion_approximation import (
    _numba_update_channels,
)


@pytest.fixture
def c() -> np.ndarray:
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture
def propensities() -> np.ndarray:
    return np.ones((3, 3))


@pytest.fixture
def boundary_thresholds() -> np.ndarray:
    return np.zeros((3, 3))


@pytest.fixture
def jump_channels() -> np.ndarray:
    return np.zeros((3, 3), dtype=bool)


@pytest.fixture
def jump_thresholds() -> np.ndarray:
    return np.full((3, 3), 5.0)


@pytest.fixture
def jump_integrated_times() -> np.ndarray:
    return np.full((3, 3), 7.0)


def test_leave_jump_phase(
    c,
    propensities,
    boundary_thresholds,
    jump_channels,
    jump_thresholds,
    jump_integrated_times,
):
    jump_channels[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] = True
    _numba_update_channels(
        c,
        propensities,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        0.1,
        boundary_thresholds,
    )
    assert not np.any(jump_channels)
    assert np.all(jump_thresholds == 5.0)
    assert np.all(jump_integrated_times == 7.0)


@pytest.mark.parametrize(
    "channel,propensity",
    [
        ((0, 2), 0.05),
        ((1, 0), 0.1),
        ((2, 1), 0.0),
    ],
)
def test_start_jump_phase(
    c,
    propensities,
    boundary_thresholds,
    jump_channels,
    jump_thresholds,
    jump_integrated_times,
    channel,
    propensity,
):
    propensities[channel] = propensity
    _numba_update_channels(
        c,
        propensities,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        0.1,
        boundary_thresholds,
    )
    expected_channels = np.zeros((3, 3), dtype=bool)
    expected_channels[channel] = True
    assert np.all(jump_channels == expected_channels)

    # new jump phase: integrated time is reset and threshold is resampled
    assert jump_integrated_times[channel] == 0
    assert jump_thresholds[channel] > 0
    assert jump_thresholds[channel] != 5.0

    # all other entries are untouched
    other = ~expected_channels
    assert np.all(jump_integrated_times[other] == 7.0)
    assert np.all(jump_thresholds[other] == 5.0)


def test_continue_jump_phase(
    c,
    propensities,
    boundary_thresholds,
    jump_channels,
    jump_thresholds,
    jump_integrated_times,
):
    propensities[:] = 0
    jump_channels[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] = True
    old_channels = jump_channels.copy()
    _numba_update_channels(
        c,
        propensities,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        0.1,
        boundary_thresholds,
    )
    assert np.all(jump_channels == old_channels)
    assert np.all(jump_thresholds == 5.0)
    assert np.all(jump_integrated_times == 7.0)


@pytest.mark.parametrize(
    "boundary_index,boundary,expected_channels",
    [
        ((0, 1), 0.5, [(0, 1), (1, 0)]),
        ((0, 1), 0.35, []),  # c[1] <= 0.35 but boundary[0, 1] belongs to c[0]
        ((2, 1), 0.25, [(2, 1), (1, 2)]),
        ((1, 0), 0.3, [(1, 0), (0, 1)]),
        ((2, 0), 0.1, []),
    ],
)
def test_boundary_thresholds(
    c,
    propensities,
    boundary_thresholds,
    jump_channels,
    jump_thresholds,
    jump_integrated_times,
    boundary_index,
    boundary,
    expected_channels,
):
    boundary_thresholds[boundary_index] = boundary
    _numba_update_channels(
        c,
        propensities,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        0.1,
        boundary_thresholds,
    )
    expected = np.zeros((3, 3), dtype=bool)
    for channel in expected_channels:
        expected[channel] = True
    assert np.all(jump_channels == expected)


def test_new_thresholds_exponentially_distributed():
    num_states = 60
    c = np.full(num_states, 1 / num_states)
    propensities = np.zeros((num_states, num_states))
    boundary_thresholds = np.zeros((num_states, num_states))
    jump_channels = np.zeros((num_states, num_states), dtype=bool)
    jump_thresholds = np.zeros((num_states, num_states))
    jump_integrated_times = np.ones((num_states, num_states))

    _numba_update_channels(
        c,
        propensities,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        0.1,
        boundary_thresholds,
    )
    assert np.sum(jump_channels) == num_states * (num_states - 1)
    assert np.all(jump_thresholds[jump_channels] > 0)
    assert np.isclose(np.mean(jump_thresholds[jump_channels]), 1.0, atol=0.1)
    assert np.all(jump_integrated_times[jump_channels] == 0)
