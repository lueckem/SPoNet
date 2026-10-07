import numpy as np
import pytest
from numba import njit

from sponet.cnvm.approximations.hybrid_jump_diffusion_approximation import (
    _numba_compute_timestep,
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


@pytest.fixture
def rates() -> tuple[np.ndarray, np.ndarray]:
    r = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 0.5, 0.0]])
    r_tilde = np.array([[0.0, 0.2, 0.1], [0.1, 0.0, 0.1], [0.1, 0.2, 0.0]])
    return r, r_tilde


def _expected_propensities(c, r, r_tilde):
    props = c[:, None] * (r * c[None, :] + r_tilde)
    np.fill_diagonal(props, 0)
    return props


def test_timestep_zero_propensities_no_change(
    c, jump_channels, jump_thresholds, jump_integrated_times, rates
):
    r, r_tilde = rates
    c_old = c.copy()
    _numba_compute_timestep(
        c,
        np.zeros(3),
        np.zeros((3, 3)),
        0.1,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        r,
        r_tilde,
        10,
    )
    assert np.allclose(c, c_old)


def test_timestep_updates_propensities(
    c, jump_channels, jump_thresholds, jump_integrated_times, rates
):
    r, r_tilde = rates
    propensities = _expected_propensities(c, r, r_tilde)
    _numba_compute_timestep(
        c,
        np.zeros(3),
        propensities,
        0.01,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        r,
        r_tilde,
        100,
    )
    assert np.allclose(propensities, _expected_propensities(c, r, r_tilde))


def test_timestep_jump_not_fired(c, jump_thresholds, jump_integrated_times, rates):
    r, r_tilde = rates
    num_agents = 10
    delta_t = 0.1
    propensities = np.zeros((3, 3))
    propensities[0, 1] = 0.5
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_channels[0, 1] = True
    jump_integrated_times[0, 1] = 0
    c_old = c.copy()

    _numba_compute_timestep(
        c,
        np.zeros(3),
        propensities,
        delta_t,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        r,
        r_tilde,
        num_agents,
    )
    assert np.allclose(c, c_old)
    assert np.isclose(jump_integrated_times[0, 1], num_agents * 0.5 * delta_t)
    assert jump_thresholds[0, 1] == 5.0


def test_timestep_jump_fired(c, jump_thresholds, jump_integrated_times, rates):
    r, r_tilde = rates
    num_agents = 10
    propensities = np.zeros((3, 3))
    propensities[0, 1] = 0.5
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_channels[0, 1] = True
    jump_integrated_times[0, 1] = 4.9
    c_old = c.copy()

    _numba_compute_timestep(
        c,
        np.zeros(3),
        propensities,
        0.1,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        r,
        r_tilde,
        num_agents,
    )
    assert np.allclose(c, c_old + np.array([-0.1, 0.1, 0]))
    assert jump_integrated_times[0, 1] == 0
    assert jump_thresholds[0, 1] != 5.0
    assert jump_thresholds[0, 1] > 0


def test_timestep_jump_clipped(jump_thresholds, jump_integrated_times, rates):
    r, r_tilde = rates
    c = np.array([0.05, 0.45, 0.5])
    propensities = np.zeros((3, 3))
    propensities[0, 2] = 1.0
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_channels[0, 2] = True

    _numba_compute_timestep(
        c,
        np.zeros(3),
        propensities,
        1.0,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        r,
        r_tilde,
        10,
    )
    assert np.allclose(c, [0, 0.45, 0.55])


def test_timestep_diffusion_stays_in_simplex(rates):
    r, r_tilde = rates
    for _ in range(200):
        c = np.array([0.01, 0.01, 0.98])
        propensities = _expected_propensities(c, r, r_tilde)
        _numba_compute_timestep(
            c,
            np.zeros(3),
            propensities,
            0.5,
            np.zeros((3, 3), dtype=bool),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            r,
            r_tilde,
            10,
        )
        assert np.all(c >= 0)
        assert np.isclose(np.sum(c), 1)


@njit()
def _seed_numba(seed):
    np.random.seed(seed)


def test_timestep_jump_channels_do_not_diffuse(c, rates):
    r, r_tilde = rates
    c_old = c.copy()
    propensities = _expected_propensities(c, r, r_tilde)
    jump_channels = ~np.eye(3, dtype=bool)
    jump_thresholds = np.full((3, 3), np.inf)
    jump_integrated_times = np.zeros((3, 3))

    _numba_compute_timestep(
        c,
        np.zeros(3),
        propensities,
        0.1,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        r,
        r_tilde,
        10,
    )
    assert np.all(c == c_old)
    assert np.allclose(
        jump_integrated_times, 10 * 0.1 * _expected_propensities(c_old, r, r_tilde)
    )


def test_timestep_jump_fires_at_threshold(c, rates):
    r, r_tilde = rates
    propensities = np.zeros((3, 3))
    propensities[1, 2] = 0.5
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_channels[1, 2] = True
    jump_thresholds = np.full((3, 3), 0.5)
    jump_integrated_times = np.zeros((3, 3))
    c_old = c.copy()

    # integrated time = 10 * 0.5 * 0.1 = 0.5 = threshold
    _numba_compute_timestep(
        c,
        np.zeros(3),
        propensities,
        0.1,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        r,
        r_tilde,
        10,
    )
    assert np.allclose(c, c_old + np.array([0, -0.1, 0.1]))
    assert jump_integrated_times[1, 2] == 0


def test_timestep_multiple_jumps_clipped_sequentially(rates):
    r, r_tilde = rates
    c = np.array([0.05, 0.45, 0.5])
    propensities = np.zeros((3, 3))
    propensities[0, 1] = 1.0
    propensities[0, 2] = 1.0
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_channels[0, 1] = True
    jump_channels[0, 2] = True
    jump_thresholds = np.full((3, 3), 5.0)
    jump_integrated_times = np.zeros((3, 3))

    _numba_compute_timestep(
        c,
        np.zeros(3),
        propensities,
        1.0,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        r,
        r_tilde,
        10,
    )
    # channel (0, 1) is processed first and takes the remaining mass of opinion 0
    assert np.allclose(c, [0, 0.5, 0.5])
    # both channels fired, so both are reset
    assert jump_integrated_times[0, 1] == 0
    assert jump_integrated_times[0, 2] == 0
    assert jump_thresholds[0, 1] != 5.0
    assert jump_thresholds[0, 2] != 5.0


def test_timestep_c_buf_holds_new_state(c, rates):
    r, r_tilde = rates
    c_buf = np.zeros(3)
    _numba_compute_timestep(
        c,
        c_buf,
        _expected_propensities(c, r, r_tilde),
        0.1,
        np.zeros((3, 3), dtype=bool),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        r,
        r_tilde,
        10,
    )
    assert np.all(c_buf == c)


def test_timestep_mixed_conserves_mass(rates):
    r, r_tilde = rates
    _seed_numba(1)
    rng = np.random.default_rng(1)
    num_agents = 20
    c = np.array([0.1, 0.3, 0.6])
    c_buf = np.zeros(3)
    propensities = _expected_propensities(c, r, r_tilde)
    jump_thresholds = rng.exponential(size=(3, 3))
    jump_integrated_times = np.zeros((3, 3))
    for _ in range(2000):
        jump_channels = rng.random((3, 3)) < 0.5
        _numba_compute_timestep(
            c,
            c_buf,
            propensities,
            0.05,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            r,
            r_tilde,
            num_agents,
        )
        assert np.all(c >= 0)
        assert np.isclose(np.sum(c), 1)


def test_timestep_diffusion_moments(rates):
    """
    Increments of a pure diffusion step have mean drift * delta_t
    and covariance delta_t / num_agents * sum_k nu_k nu_k^T a_k.
    """
    r, r_tilde = rates
    _seed_numba(2)
    num_agents = 100
    delta_t = 0.01
    num_samples = 20000
    c0 = np.array([0.3, 0.3, 0.4])
    props0 = _expected_propensities(c0, r, r_tilde)

    increments = np.zeros((num_samples, 3))
    c = np.zeros(3)
    c_buf = np.zeros(3)
    propensities = np.zeros((3, 3))
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_thresholds = np.zeros((3, 3))
    jump_integrated_times = np.zeros((3, 3))
    for k in range(num_samples):
        c[:] = c0
        propensities[:] = props0
        _numba_compute_timestep(
            c,
            c_buf,
            propensities,
            delta_t,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            r,
            r_tilde,
            num_agents,
        )
        increments[k] = c - c0

    drift = props0.sum(axis=0) - props0.sum(axis=1)
    cov = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            nu = np.zeros(3)
            nu[i], nu[j] = -1, 1
            cov += np.outer(nu, nu) * props0[i, j]
    cov *= delta_t / num_agents

    std_of_mean = np.sqrt(np.diag(cov) / num_samples)
    assert np.all(np.abs(increments.mean(axis=0) - drift * delta_t) < 5 * std_of_mean)
    assert np.allclose(np.cov(increments.T), cov, rtol=0.05, atol=1e-7)
