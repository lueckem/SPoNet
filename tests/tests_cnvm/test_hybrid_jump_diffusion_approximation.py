import numpy as np
import pytest
from numba import njit

from sponet.cnvm.approximations.hybrid_jump_diffusion_approximation import (
    _numba_compute_timestep,
    _numba_jda,
    _numba_update_channels,
    _numba_update_propensities,
)

# ---------- tests _numba_update_channels ---------------


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


@pytest.fixture
def rates() -> tuple[np.ndarray, np.ndarray]:
    r = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 0.5, 0.0]])
    r_tilde = np.array([[0.0, 0.2, 0.1], [0.1, 0.0, 0.1], [0.1, 0.2, 0.0]])
    return r, r_tilde


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
    # all channels are jump and remain jump
    propensities[:] = 0
    jump_channels[:] = True
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
    c,  # [0.5, 0.3, 0.2]
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


# ---------- tests _numba_update_propensities ---------------


def _expected_propensities(c, r, r_tilde):
    props = c[:, None] * (r * c[None, :] + r_tilde)
    np.fill_diagonal(props, 0)
    return props


def test_update_propensities(c, rates):
    r, r_tilde = rates
    propensities = np.zeros((3, 3))
    _numba_update_propensities(propensities, c, r, r_tilde)
    assert np.allclose(propensities, _expected_propensities(c, r, r_tilde))


# ---------- tests _numba_compute_timestep ---------------


def test_timestep_zero_propensities_no_change(
    c, jump_channels, jump_thresholds, jump_integrated_times, rates
):
    r, r_tilde = rates
    c_old = c.copy()
    _numba_compute_timestep(
        c,
        np.zeros((3, 3)),
        0.1,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        10,
        np.zeros((3, 3), dtype=np.int64),
    )
    assert np.allclose(c, c_old)


def test_timestep_does_not_change_propensities(
    c, jump_channels, jump_thresholds, jump_integrated_times, rates
):
    r, r_tilde = rates
    propensities = _expected_propensities(c, r, r_tilde)
    old_propensities = propensities.copy()
    _numba_compute_timestep(
        c,
        propensities,
        0.01,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        100,
        np.zeros((3, 3), dtype=np.int64),
    )
    assert np.all(propensities == old_propensities)


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
        propensities,
        delta_t,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        num_agents,
        np.zeros((3, 3), dtype=np.int64),
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
    # integrated time exceeds the threshold 5.0 only marginally, so the carried-over
    # excess is negligible and a second jump is practically impossible
    jump_integrated_times[0, 1] = 4.5 + 1e-9
    c_old = c.copy()

    _numba_compute_timestep(
        c,
        propensities,
        0.1,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        num_agents,
        np.zeros((3, 3), dtype=np.int64),
    )
    assert np.allclose(c, c_old + np.array([-0.1, 0.1, 0]))
    assert np.isclose(jump_integrated_times[0, 1], 0, atol=1e-8)
    assert jump_thresholds[0, 1] != 5.0
    assert jump_thresholds[0, 1] > 0


def test_timestep_jump_clipped(jump_thresholds, jump_integrated_times, rates):
    r, r_tilde = rates
    c = np.array([0.05, 0.45, 0.5])
    propensities = np.zeros((3, 3))
    propensities[0, 2] = 1.0
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_channels[0, 2] = True
    jump_integrated_times[0, 2] = 0

    _numba_compute_timestep(
        c,
        propensities,
        1.0,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        10,
        np.zeros((3, 3), dtype=np.int64),
    )
    assert np.allclose(c, [0, 0.45, 0.55])


def test_timestep_diffusion_stays_in_simplex(rates):
    r, r_tilde = rates
    for _ in range(200):
        c = np.array([0.01, 0.01, 0.98])
        propensities = _expected_propensities(c, r, r_tilde)
        _numba_compute_timestep(
            c,
            propensities,
            0.5,
            np.zeros((3, 3), dtype=bool),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            10,
            np.zeros((3, 3), dtype=np.int64),
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
        propensities,
        0.1,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        10,
        np.zeros((3, 3), dtype=np.int64),
    )
    assert np.all(c == c_old)
    assert np.allclose(
        jump_integrated_times, 10 * 0.1 * _expected_propensities(c_old, r, r_tilde)
    )


def test_timestep_jump_not_fired_at_threshold(c, rates):
    r, r_tilde = rates
    propensities = np.zeros((3, 3))
    propensities[1, 2] = 0.5
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_channels[1, 2] = True
    jump_thresholds = np.full((3, 3), 0.5)
    jump_integrated_times = np.zeros((3, 3))
    c_old = c.copy()

    # integrated time = 10 * 0.5 * 0.1 = 0.5 = threshold, which is not exceeded
    _numba_compute_timestep(
        c,
        propensities,
        0.1,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        10,
        np.zeros((3, 3), dtype=np.int64),
    )
    assert np.all(c == c_old)
    assert np.isclose(jump_integrated_times[1, 2], 0.5)
    assert jump_thresholds[1, 2] == 0.5


def test_timestep_zero_propensity_never_jumps():
    """
    A jump channel with zero propensity must not fire, even with a zero threshold.
    Voter model with extinct opinion 1: it must stay extinct.
    """
    r = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    r_tilde = np.zeros((3, 3))
    c = np.array([0.5, 0.0, 0.5])
    propensities = _expected_propensities(c, r, r_tilde)
    jump_channels = ~np.eye(3, dtype=bool)

    _numba_compute_timestep(
        c,
        propensities,
        0.1,
        jump_channels,
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        10,
        np.zeros((3, 3), dtype=np.int64),
    )
    assert c[1] == 0


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
        propensities,
        1.0,
        jump_channels,
        jump_thresholds,
        jump_integrated_times,
        10,
        np.zeros((3, 3), dtype=np.int64),
    )
    # channel (0, 1) is processed first and takes the remaining mass of opinion 0
    assert np.allclose(c, [0, 0.5, 0.5])
    # both channels fired, so both have new thresholds that are not yet exceeded
    assert jump_integrated_times[0, 1] < jump_thresholds[0, 1]
    assert jump_integrated_times[0, 2] < jump_thresholds[0, 2]
    assert jump_thresholds[0, 1] != 5.0
    assert jump_thresholds[0, 2] != 5.0


def test_timestep_mixed_conserves_mass(rates):
    r, r_tilde = rates
    _seed_numba(1)
    rng = np.random.default_rng(1)
    num_agents = 20
    c = np.array([0.1, 0.3, 0.6])
    propensities = _expected_propensities(c, r, r_tilde)
    jump_thresholds = rng.exponential(size=(3, 3))
    jump_integrated_times = np.zeros((3, 3))
    for _ in range(2000):
        jump_channels = rng.random((3, 3)) < 0.5
        _numba_compute_timestep(
            c,
            propensities,
            0.05,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            num_agents,
            np.zeros((3, 3), dtype=np.int64),
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
    propensities = np.zeros((3, 3))
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_thresholds = np.zeros((3, 3))
    jump_integrated_times = np.zeros((3, 3))
    for k in range(num_samples):
        c[:] = c0
        propensities[:] = props0
        _numba_compute_timestep(
            c,
            propensities,
            delta_t,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            num_agents,
            np.zeros((3, 3), dtype=np.int64),
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


@pytest.mark.parametrize("jumps_per_step", [0.5, 2.0])
def test_timestep_jump_count_poisson(rates, jumps_per_step):
    """
    With fixed propensity the number of jumps per step is Poisson distributed
    with mean num_agents * propensity * delta_t, which requires carrying over
    the excess integrated time and allowing multiple jumps per step.
    """
    r, r_tilde = rates
    _seed_numba(3)
    num_agents = 100
    delta_t = 0.1
    num_steps = 20000
    c0 = np.array([0.5, 0.5, 0.0])
    propensity = jumps_per_step / (num_agents * delta_t)

    c = np.zeros(3)
    propensities = np.zeros((3, 3))
    jump_channels = np.zeros((3, 3), dtype=bool)
    jump_channels[0, 1] = True
    jump_thresholds = np.full((3, 3), np.random.default_rng(3).exponential())
    jump_integrated_times = np.zeros((3, 3))
    num_jumps = np.zeros(num_steps)
    for k in range(num_steps):
        c[:] = c0
        propensities[:] = 0
        propensities[0, 1] = propensity
        _numba_compute_timestep(
            c,
            propensities,
            delta_t,
            jump_channels,
            jump_thresholds,
            jump_integrated_times,
            num_agents,
            np.zeros((3, 3), dtype=np.int64),
        )
        num_jumps[k] = np.round((c0[0] - c[0]) * num_agents)

    std_of_mean = np.sqrt(jumps_per_step / num_steps)
    assert abs(num_jumps.mean() - jumps_per_step) < 5 * std_of_mean
    assert np.isclose(num_jumps.var(), jumps_per_step, rtol=0.1)


def test_timestep_diffusion_clipping_only_affects_channel(rates):
    """
    Clipping a diffusion increment must not change opinions
    that are not involved in the channel.
    """
    r, r_tilde = rates
    _seed_numba(4)
    jump_channels = ~np.eye(3, dtype=bool)
    jump_channels[0, 1] = False
    clipped = False
    for _ in range(100):
        c = np.array([0.001, 0.5, 0.499])
        propensities = np.zeros((3, 3))
        propensities[0, 1] = 1.0
        _numba_compute_timestep(
            c,
            propensities,
            1.0,
            jump_channels,
            np.full((3, 3), np.inf),
            np.zeros((3, 3)),
            10,
            np.zeros((3, 3), dtype=np.int64),
        )
        assert c[2] == 0.499
        assert np.all(c >= 0)
        assert np.isclose(np.sum(c), 1)
        clipped |= c[0] == 0
    assert clipped


# ----------- tests _numba_jda ------------------


def _run_jda(c_init, delta_t, t_eval, num_agents, rates, seed):
    r, r_tilde = rates
    _seed_numba(seed)
    return _numba_jda(
        np.array(c_init, dtype=float),
        delta_t,
        np.array(t_eval, dtype=float),
        num_agents,
        r,
        r_tilde,
        0.01,
        np.full((3, 3), 2 / num_agents),
    )[0]


@pytest.mark.parametrize(
    "delta_t, t_eval",
    [
        (0.01, np.linspace(0, 5, 51)),
        (0.5, np.linspace(0, 5, 51)),  # delta_t larger than the store interval
        (0.01, [0, 0.33, 2.5, 4.999]),
    ],
)
def test_jda_shape_and_simplex(rates, delta_t, t_eval):
    c_init = [0.1, 0.3, 0.6]
    c = _run_jda(c_init, delta_t, t_eval, 30, rates, 0)
    assert c.shape == (len(t_eval), 3)
    assert np.all(c[0] == c_init)
    assert np.all(c >= 0)
    assert np.allclose(np.sum(c, axis=1), 1)
    # the state actually evolves
    assert not np.allclose(c[-1], c_init)


def test_jda_seed(rates):
    t_eval = np.linspace(0, 5, 51)
    c1 = _run_jda([0.1, 0.3, 0.6], 0.01, t_eval, 30, rates, 5)
    c2 = _run_jda([0.1, 0.3, 0.6], 0.01, t_eval, 30, rates, 5)
    c3 = _run_jda([0.1, 0.3, 0.6], 0.01, t_eval, 30, rates, 6)
    assert np.all(c1 == c2)
    assert not np.all(c1 == c3)


def test_jda_extinct_opinion_stays_extinct():
    """Voter model: an extinct opinion can never reappear."""
    r = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    r_tilde = np.zeros((3, 3))
    c = _run_jda([0.5, 0.0, 0.5], 0.01, np.linspace(0, 5, 51), 30, (r, r_tilde), 0)
    assert np.all(c[:, 1] == 0)


# ----------- tests stats ------------------


def _sample_with_stats(rates, switch_propensity_threshold, boundary_threshold, t_max):
    from sponet import CNVMParameters
    from sponet.cnvm.approximations import sample_hybrid_jump_diffusion

    r, r_tilde = rates
    params = CNVMParameters(num_opinions=3, num_agents=30, r=r, r_tilde=r_tilde)
    return sample_hybrid_jump_diffusion(
        params,
        np.array([0.1, 0.3, 0.6]),
        t_max,
        20,
        switch_propensity_threshold,
        np.full((3, 3), boundary_threshold),
        delta_t=0.01,
        seed=1,
        return_channel_stats=True,
    )


def test_channel_stats_all_jump(rates):
    t_max = 2.0
    t, c, jump_times, jump_counts = _sample_with_stats(rates, np.inf, 0.0, t_max)
    assert c.shape == (20, len(t), 3)
    assert jump_times.shape == (20, 3, 3)
    assert jump_counts.shape == (20, 3, 3)
    off_diag = ~np.eye(3, dtype=bool)
    assert np.allclose(jump_times[:, off_diag], t_max)
    assert np.all(jump_times[:, ~off_diag] == 0)
    assert np.all(jump_counts[:, ~off_diag] == 0)
    assert np.all(jump_counts[:, off_diag] >= 0)
    assert np.sum(jump_counts) > 0


def test_channel_stats_all_diffusion(rates):
    _, _, jump_times, jump_counts = _sample_with_stats(rates, -1.0, -1.0, 2.0)
    assert np.all(jump_times == 0)
    assert np.all(jump_counts == 0)


def test_channel_stats_mixed(rates):
    t_max = 2.0
    _, _, jump_times, jump_counts = _sample_with_stats(rates, 0.01, 2 / 30, t_max)
    assert np.all(jump_times >= 0)
    assert np.all(jump_times <= t_max + 1e-9)
    assert np.all(jump_counts >= 0)
    # a channel without time in jump mode cannot have fired jumps
    assert np.all(jump_counts[jump_times == 0] == 0)
    assert np.any(jump_times > 0) and np.any(jump_times < t_max)


def test_channel_stats_not_returned_by_default(rates):
    from sponet import CNVMParameters
    from sponet.cnvm.approximations import sample_hybrid_jump_diffusion

    r, r_tilde = rates
    params = CNVMParameters(num_opinions=3, num_agents=30, r=r, r_tilde=r_tilde)
    out = sample_hybrid_jump_diffusion(
        params,
        [0.1, 0.3, 0.6],
        1.0,
        5,
        0.01,
        np.full((3, 3), 0.1),
        delta_t=0.01,
        seed=1,
    )
    assert len(out) == 2
