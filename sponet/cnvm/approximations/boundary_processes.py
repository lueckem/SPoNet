import numpy as np
from numpy.typing import NDArray

from numba import njit


@njit(cache=True)
def clip_boundary(
		x: NDArray,
		t: NDArray,
		timestep_index: int,
		_max_time: float,
		_num_time_steps: int,
		_num_agents: int,
		_r: NDArray,
		_r_tilde: NDArray,
) -> tuple[NDArray, NDArray, int]:
	x[timestep_index] = np.clip(x[timestep_index], -1, 1)
	x[timestep_index] /= np.sum(x[timestep_index])

	return x, t, timestep_index


@njit(cache=True)
def _simulate_boundary_jump_process(
		t_eval: NDArray,
		x_store: NDArray,
		t_before_breach: float,
		t_after_breach: float,
		state_before_breach: NDArray,
		state_after_breach: NDArray,
		next_save_index: int,
		n_nodes: int,
		r: NDArray,
		r_tilde: NDArray,
) -> tuple[NDArray, float, NDArray, int]:
	"""
	Simulates a jump process in the boundary until it jumps outside the boundary.

	Parameters
	----------
	t_eval: NDArray
		Shape = (num_timesteps,)
	x_store: NDArray
		Shape = (num_time_steps, n_states)
	t_before_breach: float
	t_after_breach: float
	state_before_breach: NDArray
		Shape = (n_states,)
	state_after_breach: NDArray
		Shape = (n_states,)
	next_save_index: int
	n_nodes: int
	r: NDArray
		Shape = (n_states, n_states)
	r_tilde: NDArray
		Shape = (n_states, n_states)

	Returns
	-------
	tuple[NDArray, float, NDArray, int]
		(x_store, current_t, current_share, save_index)
	"""

	n_states = x_store.shape[1]

	# If two sides are breached, the line between before_breach and after_breach will a.s. not hit the corner
	breached_side_index = int(np.argmin(state_after_breach))
	initial_time, initial_share = _compute_intersection_with_boundary(
		breached_side_index,
		state_before_breach,
		state_after_breach,
		t_before_breach,
		t_after_breach,
		n_states
	)

	propensities = np.zeros((n_states, n_states))
	current_t = initial_time
	current_shares = initial_share

	while True:
		_update_boundary_propensities(propensities, r, r_tilde, current_shares, n_nodes)

		sum_props = np.sum(propensities)
		current_t += np.random.exponential(1 / sum_props)
		cum_sum = np.cumsum(propensities / sum_props)
		reaction = np.searchsorted(cum_sum, np.random.random(), side="right")
		m, n = reaction // n_states, reaction % n_states

		while current_t > t_eval[next_save_index]:
			x_store[next_save_index] = current_shares
			next_save_index += 1
			if next_save_index == len(t_eval):
				return x_store, current_t, current_shares, next_save_index

		current_shares[m] -= 1 / n_nodes
		current_shares[n] += 1 / n_nodes

		# Check if still in boundary
		if not (current_shares <= 0).any():
			return x_store, current_t, current_shares, next_save_index



@njit(cache=True)
def _compute_intersection_with_boundary(
		breached_side_index: int,
		state_before_breach: NDArray,
		state_after_breach: NDArray,
		t_before_breach: float,
		t_after_breach: float,
		n_states: int
) -> tuple[float, NDArray]:

	normal_vec = np.ones(n_states)
	normal_vec[breached_side_index] -= n_states
	s_star = (1 - normal_vec @ state_before_breach) / (
			normal_vec.transpose() @ (state_after_breach - state_before_breach)
	)
	initial_time = float(t_before_breach + s_star * (t_after_breach - t_before_breach))
	initial_share = state_before_breach + s_star * (state_after_breach - state_before_breach)
	return initial_time, initial_share


@njit(cache=True)
def _update_boundary_propensities(
		props: NDArray,
		r: NDArray,
		r_tilde: NDArray,
		c: NDArray,
		n_nodes: int,
):
	for m in range(props.shape[0]):
		for n in range(props.shape[1]):
			if m == n:
				continue
			if c[m] <= 1 / n_nodes:
				props[m][n] = 0
				continue
			props[m, n] = c[m] * (r[m, n] * c[n] + r_tilde[m, n])
	props *= n_nodes
