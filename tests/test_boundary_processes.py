import pytest
import numpy as np

from sponet.cnvm.approximations import boundary_processes as bp


@pytest.mark.parametrize(
	"state_before_breach, state_after_breach",
	[
		([.2, .4, .4], [-.2, .6, .6]),
		([.2, .4, .4], [-1, 0.8, 1.2]),
		([0.1, 0.8, 0.1], [-0.2, 0.05, 1.15]),
		([.2, .4, .4], [-.1, -.1, 1.2]),
		([1/3, 1/3, 1/3], [-.5, -.5, 2])
	]
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

	x_store, current_t, current_state, index = bp.simulate_boundary_jump_process(
		t_eval=t,
		x_store=x_store,
		t_before_breach=t[0],
		t_after_breach=t[1],
		state_before_breach=x_store[0],
		state_after_breach=state_after_breach,
		next_save_index=1,
		n_nodes=n_nodes,
		r=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
		r_tilde=np.array([[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]])
	)

	assert (current_state > 0).all()
	assert (current_state == 1 / n_nodes).any()
	assert np.isclose(np.sum(current_state), 1)

	if index < n_timesteps:
		assert not np.any(x_store[index])
		assert current_t < t[index]

	if index >= n_timesteps:
		assert current_t > t[-1]


@pytest.mark.parametrize(
	"state_before_breach, state_after_breach, breached_side_index",
	[
		([.2, .4, .4], [-.2, .6, .6], 0),
		([.2, .4, .4], [-1, 0.8, 1.2], 0),
		([0.1, 0.8, 0.1], [-0.2, 0.05, 1.15], 0),
		([1/3, 1/3, 1/3], [-.5, -.5, 2], 0),
	]
)
def test_compute_intersection_with_boundary(
		state_before_breach,
		state_after_breach,
		breached_side_index
):
	state_before_breach = np.array(state_before_breach)
	state_after_breach = np.array(state_after_breach)

	n_states = 3

	intersection_time, intersection_value = bp._compute_intersection_with_boundary(
		breached_side_index, state_before_breach, state_after_breach, 0, 1, n_states
	)
	assert 0 <= intersection_time <= 1
	assert (intersection_value == 0).any()
	assert (intersection_value >= 0).all()
	assert np.isclose(np.sum(intersection_value), 1)

	assert np.allclose(
		state_before_breach + intersection_time * (state_after_breach-state_before_breach),
		intersection_value
	)


@pytest.mark.parametrize(
	"x, expected",
	[
		([.5, .5], [.5, .5]),
		([-1, 2], [0, 1]),
		([-.2, .6, .6], [0, .5, .5]),
		([-1, .8, 1.2], [0, .3, .7]),
		([.2, .2, .6], [.2, .2, .6]),
		([2, -1, -1], [1, 0, 0])
	]
)
def test_project_onto_standard_simplex(x, expected):
	x = np.array(x)
	expected = np.array(expected)
	assert np.allclose(expected, bp._project_onto_standard_simplex(x))



