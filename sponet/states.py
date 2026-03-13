import time
from collections.abc import Callable
from warnings import warn

import networkx as nx
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray

from .collective_variables import CollectiveVariable
from .sampling import sample_randint_other
from .utils import counts_from_shares


def _sample_states(
    func: Callable[[int], NDArray], num_states: int, unique: bool
) -> NDArray:
    """
    The sampling function `func` should accept a number n
    and then return an array of samples with shape (n, d).
    This function calls `func` to generate `num_states` samples.
    If unique is True, it deletes duplicates and calls `func` again,
    until the requested `num_states` are reached.
    Also, if `num_states` = 1, it squeezes the additional dimension.
    """
    states = func(num_states)
    if states.shape[0] == 1:  # squeeze
        return states[0]

    if not unique:
        return states

    # make unique
    states = np.unique(states, axis=0)
    while states.shape[0] < num_states:
        num_missing = num_states - states.shape[0]
        states = np.concatenate([states, func(num_missing)])
        states = np.unique(states, axis=0)
        if num_states - states.shape[0] == num_missing:
            warn("Sampling of unique states might be in an endless loop.")

    return states


def sample_states_uniform(
    num_agents: int,
    num_opinions: int,
    num_states: int = 1,
    rng: Generator = default_rng(),
    unique: bool = True,
) -> NDArray:
    """
    Sample uniformly random states.

    In each state, each agent's opinion is uniform in {0, ..., num_opinions - 1}.

    Parameters
    ----------
    num_agents : int
    num_opinions : int
    num_states : int, optional
        Default: 1.
    rng : Generator, optional
        Random number generator.
    unique : bool, optional
        Whether states should be unique. Default: True.

    Returns
    -------
    NDArray
        shape = (num_states, num_agents) or shape = (num_agents,) if num_states = 1.
    """

    def _sample(n: int) -> NDArray:
        return rng.integers(num_opinions, size=(n, num_agents))

    return _sample_states(_sample, num_states, unique)


def sample_states_uniform_shares(
    num_agents: int,
    num_opinions: int,
    num_states: int = 1,
    rng: Generator = default_rng(),
    unique: bool = True,
) -> NDArray:
    """
    Sample random states with uniform opinion shares.

    The states are such that the shares of each opinion are uniform
    on the simplex of opinion shares.
    Each state is randomly shuffled.

    Parameters
    ----------
    num_agents : int
    num_opinions : int
    num_states : int, optional
        Default: 1.
    rng : Generator, optional
        Random number generator.
    unique : bool, optional
        Whether states should be unique. Default: True.

    Returns
    -------
    NDArray
        shape = (num_states, num_agents) or shape = (num_agents,) if num_states = 1.
    """
    alpha = np.ones(num_opinions)
    opinion_indices = np.arange(num_opinions)

    def _sample(n: int) -> NDArray:
        x = np.zeros((n, num_agents), dtype=int)
        shares = rng.dirichlet(alpha, n)
        counts = counts_from_shares(shares, num_agents)
        for i in range(n):
            x[i, :] = np.repeat(opinion_indices, counts[i])
        rng.shuffle(x, axis=1)
        return x

    return _sample_states(_sample, num_states, unique)


def sample_states_target_shares(
    num_agents: int,
    target_shares: ArrayLike,
    num_states: int = 1,
    rng: Generator = default_rng(),
    unique: bool = True,
) -> NDArray:
    """
    Sample random states with target opinion shares.

    Each state respects the given target_shares of each opinion and is randomly shuffled.
    The target_shares have to be non-negative with sum(target_shares) = 1.

    Parameters
    ----------
    num_agents : int
    target_shares : ArrayLike
        shape = (num_opinions,)
    num_states : int, optional
        Default: 1.
    rng : Generator, optional
        Random number Generator.
    unique : bool, optional
        Whether states should be unique. Default: True.

    Returns
    -------
    NDArray
        shape = (num_states, num_agents) or shape = (num_agents,) if num_states = 1.
    """
    target_counts = counts_from_shares(target_shares, num_agents)
    num_opinions = target_counts.shape[0]
    x_ordered = np.repeat(np.arange(num_opinions), target_counts)

    def _sample(n: int) -> NDArray:
        x = np.tile(x_ordered, (n, 1))
        rng.shuffle(x, axis=1)
        return x

    return _sample_states(_sample, num_states, unique)


def sample_states_local_clusters(
    network: nx.Graph,
    num_opinions: int,
    num_states: int = 1,
    max_num_seeds: int = 1,
    min_num_seeds: int = 1,
    rng: Generator = default_rng(),
    unique: bool = True,
) -> NDArray:
    """
    Create states by the following procedure:
    1) Pick uniformly random opinion shares
    2) Pick num_seeds random seeds on the graph for each opinion
    (num_seeds is uniformly random between min_num_seeds and max_num_seeds)
    3) Propagate the opinions outward from each seed to neighboring nodes until the shares are reached

    Parameters
    ----------
    network : nx.Graph
    num_opinions : int
    num_states : int, optional
        Default: 1.
    max_num_seeds : int, optional
    min_num_seeds : int, optional
    rng : Generator, optional
        random number generator
    unique : bool, optional
        Whether states should be unique. Default: True.

    Returns
    -------
    NDArray
        shape = (num_states, num_agents) or shape = (num_agents,) if num_states = 1.
    """
    num_agents = network.number_of_nodes()
    alpha = np.ones(num_opinions)

    def _sample(n: int) -> NDArray:
        x = np.zeros((n, num_agents), dtype=int)
        target_shares = rng.dirichlet(alpha, n)
        target_counts = counts_from_shares(target_shares, num_agents)
        num_seeds = rng.integers(min_num_seeds, max_num_seeds + 1, size=n)

        for i in range(n):
            seeds = rng.choice(
                num_agents, size=num_seeds[i] * num_opinions, replace=False
            )
            rng.shuffle(seeds)
            seeds = list(seeds.reshape((num_opinions, num_seeds[i])))
            x[i] = _state_local_clusters(
                target_counts[i],
                seeds,
                network,
                rng,
            )
        return x

    return _sample_states(_sample, num_states, unique)


def _state_local_clusters(
    target_counts: NDArray,
    seeds: list[NDArray],
    network: nx.Graph,
    rng: Generator,
) -> NDArray:
    num_agents = network.number_of_nodes()
    num_opinions = target_counts.shape[0]
    counts_reached = np.full(num_opinions, fill_value=False)
    x = np.full(num_agents, fill_value=-1, dtype=int)  # -1 stands for not yet specified
    counts = np.zeros(num_opinions, dtype=int)  # current counts for each opinion
    opinions = np.arange(num_opinions)

    while True:
        rng.shuffle(opinions)

        # iterate through seeds and propagate opinions
        for m in opinions:
            # if counts are reached, there is nothing to do
            if counts_reached[m]:
                continue

            # if there are no seeds available, add a random new one
            if len(seeds[m]) == 0:
                possible_idx = np.nonzero(x == -1)[0]
                seeds[m] = np.array([rng.choice(possible_idx)])

            new_seeds_m = []
            # set opinion of seeds to m
            for seed in seeds[m]:
                if x[seed] != -1:
                    continue

                if counts[m] < target_counts[m]:
                    x[seed] = m
                    counts[m] += 1
                    # add neighbors that are available as new seeds
                    new_seeds_m += [n for n in network.neighbors(seed) if x[n] == -1]

                if counts[m] == target_counts[m]:  # counts have been reached
                    counts_reached[m] = True
                    break

            new_seeds_m = np.unique(new_seeds_m)
            rng.shuffle(new_seeds_m)
            seeds[m] = new_seeds_m

        if np.all(counts_reached):
            break
    return x


def build_state_by_degree(
    network: nx.Graph,
    opinion_shares: ArrayLike,
    opinion_order: ArrayLike,
) -> NDArray:
    """
    Construct a state where the largest degree nodes have a certain opinion.

    Example
    ----------
    opinion_shares = [0.2, 0.5, 0.3], opinion_order = [1, 2, 0]
    means that the 20% of nodes with the largest degree get opinion 1,
    the subsequent 50% of nodes with the largest degree get opinion 2,
    and the remaining 30% of nodes (which will have the smallest degrees) get opinion 0.

    Parameters
    ----------
    network : nx.Graph
    opinion_shares : ArrayLike
        shape = (num_opinions,), has to sum to 1
    opinion_order : ArrayLike
        shape = (num_opinions,), permutation of {0, ..., num_opinions - 1}

    Returns
    -------
    NDArray
    """
    num_nodes = network.number_of_nodes()
    x = np.zeros(num_nodes, dtype=int)
    degrees = [d for _, d in network.degree()]  # type: ignore
    degrees_sorted_idx = np.argsort(degrees)[::-1]
    opinion_counts = counts_from_shares(opinion_shares, num_nodes)

    i = 0
    for opinion, opinion_count in zip(np.array(opinion_order), opinion_counts):
        x[degrees_sorted_idx[i : i + opinion_count]] = opinion
        i += opinion_count

    return x


# TODO: sample_states_by_degree


def sample_states_target_cvs(
    num_agents: int,
    num_opinions: int,
    col_var: CollectiveVariable,
    target_cv_value: ArrayLike,
    num_states: int = 1,
    rtol: float = 1e-4,
    rng: Generator = default_rng(),
    max_sample_time_per_state: float = 10,
    unique: bool = True,
) -> NDArray:
    """
    Sample states with target collective variable value.

    Samples a state x such that approximately
    cv(x) = target_cv_value
    via MCMC.

    Since the relative tolerance rtol is used, this function does not work
    if ||target_cv_value||=0.

    Parameters
    ----------
    num_agents : int
    num_opinions : int
    col_var : CollectiveVariable
    target_cv_value : ArrayLike
        Shape = (cv_dim,).
    num_states : int, optional
        Default: 1.
    rtol : float, optional
        Relative tolerance.
    rng : Generator, optional
        Random number generator.
    max_sample_time_per_state : float, optional
        In seconds. Raises RuntimeError if no state could be found in that time.
    unique : bool, optional
        Whether states should be unique. Default: True.

    Returns
    -------
    NDArray
        shape = (num_states, num_agents) or shape = (num_agents,) if num_states = 1.
    """
    target_cv_value = np.array(target_cv_value, ndmin=1)

    # set temperature in relation to num_agents
    base_temperature = -2.0 / num_agents / np.log(0.5)
    temperature_decay_factor = 0.1 ** (1.0 / num_agents)
    iterations_until_temperature_reset = num_agents

    def _sample(n: int) -> NDArray:
        x = np.zeros((n, num_agents), dtype=int)
        for i in range(n):
            initial_guess = _initial_guess_target_cvs(
                num_agents, num_opinions, col_var, target_cv_value, rng
            )
            x[i] = _sample_state_target_cvs(
                num_opinions,
                col_var,
                target_cv_value,
                initial_guess,
                rtol,
                base_temperature,
                temperature_decay_factor,
                iterations_until_temperature_reset,
                max_sample_time_per_state,
                rng,
            )
        return x

    return _sample_states(_sample, num_states, unique)


def _sample_state_target_cvs(
    num_opinions: int,
    cv: CollectiveVariable,
    target_cv_value: NDArray,
    initial_guess: NDArray,
    rtol: float,
    base_temperature: float,
    temperature_decay_factor: float,
    iterations_until_temperature_reset: int,
    max_sample_time: float,
    rng: Generator,
) -> NDArray:
    x = np.copy(initial_guess)

    norm_target_cv = np.linalg.norm(target_cv_value, 1)
    cv_x = cv(x)
    diff = np.linalg.norm(cv_x - target_cv_value, 1) / norm_target_cv

    num_iterations = 0
    iterations_since_improvement = 0
    temperature = base_temperature
    start = time.time()
    while diff > rtol:
        num_iterations += 1
        iterations_since_improvement += 1
        temperature *= temperature_decay_factor
        # if no improvement could be achieved, increase temperature to base again
        if iterations_since_improvement > iterations_until_temperature_reset:
            temperature = base_temperature

        opinion_to_change = num_iterations % num_opinions
        possible_idxs = np.argwhere(x == opinion_to_change)
        if len(possible_idxs) == 0:
            continue

        idx_to_change = rng.choice(possible_idxs)
        new_opinion = sample_randint_other(num_opinions, opinion_to_change, rng)
        x[idx_to_change] = new_opinion
        new_cv_x = cv(x)
        new_diff = np.linalg.norm(new_cv_x - target_cv_value, 1) / norm_target_cv
        if new_diff < diff:
            iterations_since_improvement = 0

        probability_switch = (
            1 if new_diff < diff else np.exp(-(new_diff - diff) / temperature)
        )
        if rng.random() <= probability_switch:
            diff = new_diff
        else:  # switch the opinion back
            x[idx_to_change] = opinion_to_change

        if time.time() - start > max_sample_time:
            raise RuntimeError(
                f"Timeout: could not generate a state with target CV value after {num_iterations} iterations."
            )

    return x


def _initial_guess_target_cvs(
    num_agents: int,
    num_opinions: int,
    col_var: CollectiveVariable,
    target_cv_value: NDArray,
    rng: Generator,
) -> NDArray:
    x_initial_choices = sample_states_uniform_shares(num_agents, num_opinions, 10, rng)
    cv_initial_choices = col_var(x_initial_choices)
    diffs = np.linalg.norm(cv_initial_choices - target_cv_value, 1, axis=1)
    return x_initial_choices[np.argmin(diffs)]
