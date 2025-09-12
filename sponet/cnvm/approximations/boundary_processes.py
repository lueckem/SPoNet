import numpy as np
from numpy.typing import NDArray

from numba import njit

from collections.abc import Callable


type BoundaryProcess = Callable[
	[
		NDArray,  # t_eval
		NDArray,  # x_store
		float,  # t_before_breach
		float,  # t_after_breach
		NDArray,  # state_before_breach
		NDArray,  # state_after_breach
		int,  # next_store_index
		int,  # n_nodes
		NDArray,  # r
		NDArray  # r_tilde
	],
	NDArray, float, NDArray, int, bool
]


@njit(cache=True)
def clip_to_boundary(
		_t_eval: NDArray,
		x_store: NDArray,
		_t_before_breach: float,
		t_after_breach: float,
		_state_before_breach: NDArray,
		state_after_breach: NDArray,
		next_store_index: int,
		_n_nodes: int,
		_r: NDArray,
		_r_tilde: NDArray,
) -> tuple[NDArray, float, NDArray, int, bool]:
	"""
	Cuts of any negative/larger than 1 values and renormalizes the resulting clipped vector.

	This function has no theoretical justification.
	It is easy and fast.
	Does not advance time.

	Parameters
	----------
	_t_eval: NDArray, unused
	x_store: NDArray
		Shape = (num_time_steps, n_states)
	_t_before_breach: float, unused
	t_after_breach: float
	_state_before_breach: NDArray, unused
	state_after_breach: NDArray
		Shape = (n_states,)
	next_store_index: int
	_n_nodes: int, unused
	_r: NDArray, unused
	_r_tilde: NDArray, unused

	Returns
	-------
	tuple[NDArray, float, NDArray, int, bool]
		(x_store, current_t, current_share, save_index, advances_time)
		advances_time=False
	"""
	clipped_state = np.clip(state_after_breach, 0, 1)
	clipped_state /= np.sum(clipped_state)

	return x_store, t_after_breach, clipped_state, next_store_index, False


@njit(cache=True)
def compute_normal_boundary_reflection(
		_t_eval: NDArray,
		x_store: NDArray,
		_t_before_breach: float,
		t_after_breach: float,
		_state_before_breach: NDArray,
		state_after_breach: NDArray,
		next_store_index: int,
		n_nodes: int,
		_r: NDArray,
		_r_tilde: NDArray,
) -> tuple[NDArray, float, NDArray, int, bool]:
	"""
	Computes the reflection with respect to the normal of the boundary.

	This function is well-defined for all input values.
	The best approximation of state_after_breach on the simplex is used as reflection point.
	The reflection is adapted if the formally correct reflected point is outside the boundary.
	Does not advance time.

	Parameters
	----------
	_t_eval: NDArray, unused
	x_store: NDArray
		Shape = (num_time_steps, n_states)
	_t_before_breach: float,  unused
	t_after_breach: float
	_state_before_breach: NDArray, unused
	state_after_breach: NDArray
		Shape = (n_states,)
	next_store_index: int
	n_nodes: int
	_r: NDArray, unused
	_r_tilde: NDArray, unused

	Returns
	-------
	tuple[NDArray, float, NDArray, int, bool]
		(x_store, current_t, current_share, save_index, advances_time)
		advances_time=False
	"""
	# Use projection onto simplex as reflection point.
	proj_after_breach = _project_onto_standard_simplex(state_after_breach)
	reflection = 2 * proj_after_breach - state_after_breach

	# If reflection is outside of boundary again, lower the step size
	n = 1
	while (reflection <= 0).any():
		reflection = proj_after_breach + 1 / n * (proj_after_breach - state_after_breach)
		n += 1

		if n == 10:
			# "Reflect" into the direction of the middle of the simplex.
			# This will push the point outside of corners
			n_states = state_after_breach.shape[0]
			reflection = proj_after_breach + 1 / n_nodes * (np.full(n_states, 1 / n_states) - proj_after_breach)
			break

	return x_store, t_after_breach, reflection, next_store_index, False


@njit(cache=True)
def simulate_boundary_jump_process(
		t_eval: NDArray,
		x_store: NDArray,
		t_before_breach: float,
		t_after_breach: float,
		state_before_breach: NDArray,
		state_after_breach: NDArray,
		next_store_index: int,
		n_nodes: int,
		r: NDArray,
		r_tilde: NDArray,
) -> tuple[NDArray, float, NDArray, int, bool]:
	"""
	Simulates a jump process in the boundary until it jumps outside the boundary.

	Always advances time.

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
	next_store_index: int
	n_nodes: int
	r: NDArray
		Shape = (n_states, n_states)
	r_tilde: NDArray
		Shape = (n_states, n_states)

	Returns
	-------
	tuple[NDArray, float, NDArray, int, bool]
		(x_store, current_t, current_share, save_index, advances_time)
		advances_time = True
	"""

	n_states = x_store.shape[1]

	# Compute the starting point of the boundary jump process.
	# If two sides are breached, the line between before_breach and after_breach will a.s. not hit the corner
	# This choice of starting point has no theoretical justification.
	# This choice + the design of the CLE algorithm ensures that the starting point is before the next t_eval time.
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

		while current_t >= t_eval[next_store_index]:
			x_store[next_store_index] = current_shares
			next_store_index += 1
			if next_store_index == len(t_eval):
				return x_store, current_t, current_shares, next_store_index, True

		current_shares[m] -= 1 / n_nodes
		current_shares[n] += 1 / n_nodes

		# Check if still in boundary
		if not (current_shares <= 0).any():
			return x_store, current_t, current_shares, next_store_index, True


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


@njit(cache=True)
def _project_onto_standard_simplex(x: NDArray) -> NDArray:
	"""
	Computes the best approximation of the standard d-1 simplex to a given point in R^d.

	Algorithm from https://doi.org/10.48550/arXiv.1101.6081
	Could be further sped up by using constant time median algorithm.

	Parameters
	----------
	x: NDArray

	Returns
	-------
	NDArray
	shape=x.shape

	"""
	n_states = x.shape[0]
	y = np.sort(x)
	index = n_states - 1
	while True:
		t = (np.sum(y[index:]) - 1) / (n_states - index)
		if t >= y[index - 1]:
			t_hat = t
			break
		index -= 1
		if index == 0:
			t_hat = (np.sum(y) - 1) / n_states
			break
	res = x - t_hat
	return np.where(res > 0, res, 0)


def get_boundary_process_from_alias(alias: str) -> BoundaryProcess:

	boundary_process_dict = {
		"clipping": clip_to_boundary,
		"jump": simulate_boundary_jump_process,
		"normal-reflection": compute_normal_boundary_reflection,
	}

	return boundary_process_dict[alias]
