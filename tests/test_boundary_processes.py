import pytest
import numpy as np

from sponet.cnvm.approximations.chemical_langevin_equation import (
    boundary_processes as bp,
)


@pytest.mark.parametrize(
    "state_after_breach, expected",
    [
        ([-0.2, 0.6, 0.6], [0, 0.5, 0.5]),
        ([-1, 0.8, 1.2], [0, 4 / 9, 5 / 9]),
        ([-0.1, -0.1, 1.2], [0, 0, 1]),
        ([-0.5, -0.5, 2], [0, 0, 1]),
    ],
)
def test_clip_to_boundary(state_after_breach, expected):
    state_after_breach = np.array(state_after_breach)

    n_states = state_after_breach.shape[0]
    n_nodes = 10
    n_timesteps = 10
    t_max = 10
    t = np.linspace(0, t_max, n_timesteps)
    x_store = np.zeros((n_timesteps, n_states))
    x_store[0] = np.zeros(n_states)

    new_x_store, current_t, current_state, index, _ = bp.clip_to_boundary(
        _t_eval=t,
        x_store=x_store,
        _t_before_breach=float(t[0]),
        t_after_breach=float(t[1]),
        _state_before_breach=x_store[0],
        state_after_breach=state_after_breach,
        next_store_index=1,
        _n_nodes=n_nodes,
        _r=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        _r_tilde=np.array([[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]]),
    )

    assert np.allclose(x_store, new_x_store)

    assert (current_state >= 0).all()
    assert np.isclose(np.sum(current_state), 1)

    assert np.allclose(current_state, expected)


@pytest.mark.parametrize(
    "state_after_breach, expected",
    [
        ([-0.2, 0.6, 0.6], [0.2, 0.4, 0.4]),
        ([-1, 0.8, 1.2], [0.594, 0.003, 0.403]),
        ([-0.1, -0.1, 1.2], [0.1, 0.1, 0.8]),
        ([-0.5, -0.5, 2], [0.495, 0.495, 0.01]),
        ([2, -2, 1], [0.9 + 1 / 30, 1 / 30, 1 / 30]),
    ],
)
def test_compute_normal_boundary_reflection(state_after_breach, expected):
    state_before_breach = np.array([1 / 3, 1 / 3, 1 / 3])
    state_after_breach = np.array(state_after_breach)
    expected = np.array(expected)

    n_states = state_before_breach.shape[0]
    n_nodes = 10
    n_timesteps = 10
    t_max = 10
    t = np.linspace(0, t_max, n_timesteps)
    x_store = np.zeros((n_timesteps, n_states))
    x_store[0] = state_before_breach

    new_x_store, current_t, current_state, index, _ = (
        bp.compute_normal_boundary_reflection(
            _t_eval=None,
            x_store=x_store,
            _t_before_breach=None,
            t_after_breach=float(t[1]),
            _state_before_breach=None,
            state_after_breach=state_after_breach,
            next_store_index=1,
            n_nodes=n_nodes,
            _r=None,
            _r_tilde=None,
        )
    )
    assert np.sum(current_state) == 1
    assert (current_state >= 0).all()
    assert np.allclose(current_state, expected)

    assert np.allclose(x_store, new_x_store)
    return


@pytest.mark.parametrize(
    "state_before_breach, state_after_breach",
    [
        ([0.2, 0.4, 0.4], [-0.2, 0.6, 0.6]),
        ([0.2, 0.4, 0.4], [-1, 0.8, 1.2]),
        ([0.1, 0.8, 0.1], [-0.2, 0.05, 1.15]),
        ([0.2, 0.4, 0.4], [-0.1, -0.1, 1.2]),
        ([1 / 3, 1 / 3, 1 / 3], [-0.5, -0.5, 2]),
    ],
)
def test_simulate_boundary_jump_process(state_before_breach, state_after_breach):
    state_before_breach = np.array(state_before_breach)
    state_after_breach = np.array(state_after_breach)

    n_states = state_before_breach.shape[0]
    n_nodes = 10
    n_timesteps = 10
    t_max = 10
    t = np.linspace(0, t_max, n_timesteps)
    x_store = np.zeros((n_timesteps, n_states))
    x_store[0] = state_before_breach

    x_store, current_t, current_state, index, _ = bp.simulate_boundary_jump_process(
        t_eval=t,
        x_store=x_store,
        t_before_breach=float(t[0]),
        t_after_breach=float(t[1]),
        state_before_breach=x_store[0],
        state_after_breach=state_after_breach,
        next_store_index=1,
        n_nodes=n_nodes,
        r=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        r_tilde=np.array([[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]]),
    )

    assert (current_state > 0).all()
    assert (current_state == 1 / n_nodes).any()
    assert np.isclose(np.sum(current_state), 1)

    if index < n_timesteps:
        assert not np.any(x_store[index])
        assert current_t < t[index]

    if index >= n_timesteps:
        assert current_t > t[-1]


def test_simulate_boundary_jump_process_storing():
    """
    Tests if the function correctly stores simulated values in x_store.
    """
    n_states = 3

    n_timesteps = 10
    t_max = 10
    t_eval = np.linspace(0, t_max, n_timesteps)

    x_store = np.zeros((n_timesteps, n_states))

    t_before_breach = t_eval[0] + 0.9 * (t_eval[1] - t_eval[0])
    t_after_breach = t_eval[1]
    state_before_breach = np.array([0.2, 0.4, 0.4])
    state_after_breach = np.array([-0.2, 0.6, 0.6])
    next_store_index = 1
    n_nodes = 10
    r = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    r_tilde = np.array([[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]])

    current_t = t_eval[0]
    index = 0

    while current_t < t_eval[1]:
        x_store, current_t, current_state, index, _ = bp.simulate_boundary_jump_process(
            t_eval=t_eval,
            x_store=x_store,
            t_before_breach=float(t_before_breach),
            t_after_breach=float(t_after_breach),
            state_before_breach=state_before_breach,
            state_after_breach=state_after_breach,
            next_store_index=next_store_index,
            n_nodes=n_nodes,
            r=r,
            r_tilde=r_tilde,
        )

    assert not np.allclose(x_store[next_store_index], np.zeros(n_states))
    assert index > next_store_index
    assert current_t < t_eval[index]


@pytest.mark.parametrize(
    "state_before_breach, state_after_breach, breached_side_index",
    [
        ([0.2, 0.4, 0.4], [-0.2, 0.6, 0.6], 0),
        ([0.2, 0.4, 0.4], [-1, 0.8, 1.2], 0),
        ([0.1, 0.8, 0.1], [-0.2, 0.05, 1.15], 0),
        ([1 / 3, 1 / 3, 1 / 3], [-0.5, -0.5, 2], 0),
    ],
)
def test_compute_intersection_with_boundary(
    state_before_breach, state_after_breach, breached_side_index
):
    state_before_breach = np.array(state_before_breach)
    state_after_breach = np.array(state_after_breach)

    intersection_time, intersection_value = bp._compute_intersection_with_boundary(
        breached_side_index, state_before_breach, state_after_breach, 0, 1
    )
    assert 0 <= intersection_time <= 1
    assert intersection_value[breached_side_index] == 0
    assert (intersection_value >= 0).all()
    assert np.isclose(np.sum(intersection_value), 1)

    assert np.allclose(
        state_before_breach
        + intersection_time * (state_after_breach - state_before_breach),
        intersection_value,
    )


def test_update_boundary_propensities():
    n_states = 3
    propensities = np.zeros((n_states, n_states))

    r = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    r_tilde = np.array([[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]])

    n_nodes = 10

    for i in range(n_states):
        shares = np.full(n_states, (1 - 1 / (2 * n_nodes)) / (n_states - 1))
        shares[i] = 1 / (2 * n_nodes)

        bp._update_boundary_propensities(propensities, r, r_tilde, shares, n_nodes)
        assert np.sum(propensities[i, :]) == 0


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0.5, 0.5], [0.5, 0.5]),
        ([-1, 2], [0, 1]),
        ([-0.2, 0.6, 0.6], [0, 0.5, 0.5]),
        ([-1, 0.8, 1.2], [0, 0.3, 0.7]),
        ([0.2, 0.2, 0.6], [0.2, 0.2, 0.6]),
        ([2, -1, -1], [1, 0, 0]),
    ],
)
def test_project_onto_standard_simplex(x, expected):
    x = np.array(x)
    expected = np.array(expected)
    assert np.allclose(expected, bp._project_onto_standard_simplex(x))
