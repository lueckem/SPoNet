import time

import networkx as nx
import numpy as np
from numpy.random import Generator, default_rng

from .collective_variables import CollectiveVariable


def sample_states_uniform(
    num_agents: int, num_opinions: int, num_states: int, rng: Generator = default_rng()
) -> np.ndarray:
    """
    Sample uniformly random states.

    In each state, each agent's opinion is uniform in {0, ..., num_opinions - 1}.

    Parameters
    ----------
    num_agents : int
    num_opinions : int
    num_states : int
    rng : Generator, optional
        random number generator

    Returns
    -------
    np.ndarray
        shape = (num_states, num_agents)
    """
    return rng.integers(num_opinions, size=(num_states, num_agents))


def sample_states_uniform_shares(
    num_agents: int, num_opinions: int, num_states: int, rng: Generator = default_rng()
) -> np.ndarray:
    """
    Sample random states with uniform opinion shares.

    The states are such that the shares of each opinion are uniform
    on the simplex of opinion shares.
    Each state is randomly shuffled.

    Parameters
    ----------
    num_agents : int
    num_opinions : int
    num_states : int
    rng : Generator, optional
        random number generator

    Returns
    -------
    np.ndarray
        shape = (num_states, num_agents)
    """
    x = np.zeros((num_states, num_agents))
    alpha = np.ones(num_opinions)
    for i in range(num_states):
        shares = rng.dirichlet(alpha=alpha)
        counts = np.round(shares * num_agents).astype(int)
        counts[-1] = num_agents - np.sum(counts[:-1])

        this_x = []
        for m in range(num_opinions):
            this_x += [m] * counts[m]
        rng.shuffle(this_x)
        x[i] = this_x

    x = np.unique(x.astype(int), axis=0)

    while x.shape[0] != num_states:
        missing_points = num_states - x.shape[0]
        x = np.concatenate(
            [
                x,
                sample_states_uniform_shares(
                    num_agents, num_opinions, missing_points, rng
                ),
            ]
        )
        x = np.unique(x.astype(int), axis=0)

    return x


def sample_states_target_shares(
    num_agents: int,
    target_shares: np.ndarray,
    num_states: int,
    rng: Generator = default_rng(),
) -> np.ndarray:
    """
    Sample random states with target opinion shares.

    Each state respects the given target_shares of each opinion, such that sum(target_shares) = 1.
    Each state is randomly shuffled.

    Parameters
    ----------
    num_agents : int
    target_share : np.ndarray
        shape = (num_opinions,)
    num_states : int
    rng : Generator, optional
        random number generator

    Returns
    -------
    np.ndarray
        shape = (num_states, num_agents)
    """
    x = np.zeros((num_states, num_agents)).astype(int)

    target_counts = np.round(target_shares * num_agents)
    target_counts = target_counts.astype(int)
    target_counts[-1] = max(0, num_agents - np.sum(target_counts[:-1]))
    x_ordered = np.repeat(np.arange(len(target_counts)), target_counts)

    for i in range(num_states):
        x[i] = x_ordered
        rng.shuffle(x[i])

    return x


def sample_states_local_clusters(
    network: nx.Graph,
    num_opinions: int,
    num_states: int,
    max_num_seeds: int = 1,
    min_num_seeds: int = 1,
    rng: Generator = default_rng(),
) -> np.ndarray:
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
    num_states : int
    max_num_seeds : int, optional
    min_num_seeds : int, optional
    rng : Generator, optional
        random number generator

    Returns
    -------
    np.ndarray
    """
    num_agents = network.number_of_nodes()
    x = np.zeros((num_states, num_agents))
    alpha = np.ones(num_opinions)

    for i in range(num_states):
        target_shares = rng.dirichlet(alpha=alpha)
        target_counts = np.round(target_shares * num_agents).astype(int)
        target_counts[-1] = num_agents - np.sum(target_counts[:-1])
        this_x = -1 * np.ones(num_agents)  # -1 stands for not yet specified
        counts = np.zeros(num_opinions)  # keep track of current counts for each opinion

        # pick initial seeds
        num_seeds = rng.integers(min_num_seeds, max_num_seeds + 1)
        seeds = rng.choice(num_agents, size=num_seeds * num_opinions, replace=False)
        rng.shuffle(seeds)
        seeds = seeds.reshape(
            (num_opinions, num_seeds)
        )  # keep track of seeds of each opinion
        seeds = list(seeds)

        counts_reached = np.zeros(num_opinions).astype(bool)

        while True:
            # iterate through seeds and propagate opinions
            opinions = np.array(range(num_opinions))
            rng.shuffle(opinions)
            for m in opinions:
                # if counts are reached, there is nothing to do
                if counts_reached[m]:
                    continue

                # if there are no seeds available, add a random new one
                if len(seeds[m]) == 0:
                    possible_idx = np.nonzero(this_x == -1)[0]
                    new_seed = rng.choice(possible_idx)
                    seeds[m] = np.array([new_seed])

                new_seeds_m = []
                # set opinion of seeds to m
                for seed in seeds[m]:
                    if this_x[seed] != -1:
                        continue

                    if counts[m] < target_counts[m]:
                        this_x[seed] = m
                        counts[m] += 1

                        # add neighbors that are available as new seeds
                        neighbors = np.array([n for n in network.neighbors(seed)])
                        neighbors = neighbors[this_x[neighbors] == -1]
                        new_seeds_m += neighbors.tolist()

                    if counts[m] == target_counts[m]:  # counts have been reached
                        counts_reached[m] = True
                        break

                new_seeds_m = np.unique(new_seeds_m)
                rng.shuffle(new_seeds_m)
                seeds[m] = new_seeds_m

            if np.all(counts_reached):
                break

        x[i] = this_x

    x = np.unique(x.astype(int), axis=0)

    while x.shape[0] != num_states:
        missing_points = num_states - x.shape[0]
        x = np.concatenate(
            [
                x,
                sample_states_local_clusters(
                    network,
                    num_opinions,
                    missing_points,
                    max_num_seeds,
                    min_num_seeds,
                    rng,
                ),
            ]
        )
        x = np.unique(x.astype(int), axis=0)

    return x


def build_state_by_degree(
    network: nx.Graph,
    opinion_shares: np.ndarray,
    opinion_order: np.ndarray,
) -> np.ndarray:
    """
    Construct a state where the largest degree nodes have certain opinion.

    Example
    ----------
    opinion_shares = [0.2, 0.5, 0.3], opinion_order = [1, 2, 0]
    means that the 20% of nodes with the largest degree get opinion 1,
    the subsequent 50% of nodes with the largest degree get opinion 2,
    and the remaining 30% of nodes (which will have the smallest degrees) get opinion 0.

    Parameters
    ----------
    network : nx.Graph
    opinion_shares : np.ndarray
        shape = (num_opinions,), has to sum to 1
    opinion_order : np.ndarray
        shape = (num_opinions,), permutation of {0, ..., num_opinions - 1}

    Returns
    -------
    np.ndarray
    """
    num_nodes = network.number_of_nodes()
    x = np.zeros(num_nodes, dtype=int)
    degrees = [d for _, d in network.degree()]
    degrees_sorted_idx = np.argsort(degrees)[::-1]

    opinion_counts = np.round(opinion_shares * num_nodes).astype(int)
    opinion_counts[-1] = num_nodes - np.sum(opinion_counts[:-1])

    i = 0
    for opinion, opinion_count in zip(opinion_order, opinion_counts):
        for _ in range(opinion_count):
            x[degrees_sorted_idx[i]] = opinion
            i += 1

    return x


def sample_state_target_cvs(
    num_agents: int,
    num_opinions: int,
    col_var: CollectiveVariable,
    target_cv_value: np.ndarray,
    rtol: float = 1e-4,
    rng: Generator = default_rng(),
    max_sample_time: float = 10,
) -> np.ndarray:
    """
    Sample a state with target collective variable value.

    Samples a state x such that approximately
    cv(x) = target_cv_value
    via MCMC.

    Parameters
    ----------
    num_agents : int
    num_opinions : int
    col_var : CollectiveVariable
    target_cv_value : np.ndarray
        shape = (cv_dim,)
    rtol : float, optional
        relative tolerance
    rng : Generator, optional
        random number generator
    max_sample_time : float, optional
            in seconds

    Returns
    -------
    np.ndarray
        shape = (num_agents,)
    """
    # pick the best initial state from some randomly sampled ones
    x = _initial_states_target_cvs(
        num_agents, num_opinions, 10, col_var, target_cv_value, rng
    )

    norm_target_cv = np.linalg.norm(target_cv_value, 1)
    cv_x = col_var(x[np.newaxis, :])[0]
    diff = np.linalg.norm(cv_x - target_cv_value, 1) / norm_target_cv

    # set temperature in relation to num_agents
    base_temperature = -2.0 / num_agents / np.log(0.5)
    temperature_decay_factor = 0.1 ** (1.0 / num_agents)
    temperature = base_temperature
    iterations_until_temperature_reset = num_agents

    num_iterations = 0
    iterations_since_improvement = 0
    start = time.time()
    while diff > rtol:
        num_iterations += 1
        iterations_since_improvement += 1
        temperature *= temperature_decay_factor
        # if no improvement could be achieved, increase temperature again
        if iterations_since_improvement > iterations_until_temperature_reset:
            temperature = base_temperature

        opinion_to_change = num_iterations % num_opinions
        possible_idxs = np.argwhere(x == opinion_to_change)
        if len(possible_idxs) == 0:
            continue

        idx_to_change = rng.choice(possible_idxs)
        new_opinion = rng.integers(num_opinions)
        while new_opinion == opinion_to_change:
            new_opinion = rng.integers(num_opinions)
        x[idx_to_change] = new_opinion
        new_cv_x = col_var(x[np.newaxis, :])[0]
        new_diff = np.linalg.norm(new_cv_x - target_cv_value, 1) / norm_target_cv
        if new_diff < diff:
            iterations_since_improvement = 0
        probability_switch = (
            1 if new_diff < diff else np.exp(-(new_diff - diff) / temperature)
        )
        switch = rng.random() <= probability_switch
        if switch:
            diff = new_diff
        else:
            # switch the opinion back
            x[idx_to_change] = opinion_to_change

        if time.time() - start > max_sample_time:
            raise RuntimeError(
                f"Timeout: could not generate a state with target CV value after {num_iterations} iterations."
            )

    return x


def _initial_states_target_cvs(
    num_agents: int,
    num_opinions: int,
    num_initial_samples: int,
    col_var: CollectiveVariable,
    target_cv_value: np.ndarray,
    rng: Generator,
):
    x_initial_choices = sample_states_uniform_shares(
        num_agents, num_opinions, num_initial_samples, rng
    )
    cv_initial_choices = col_var(x_initial_choices)
    diffs = [
        np.linalg.norm(cv_initial_choices[i] - target_cv_value, 1)
        for i in range(num_initial_samples)
    ]
    min_diff_idx = np.argmin(diffs)
    return x_initial_choices[min_diff_idx]
