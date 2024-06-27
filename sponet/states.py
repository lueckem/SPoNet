import numpy as np
from numpy.random import Generator, default_rng
import networkx as nx


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
    2) Pick num_seeds random seeds on the graph for each opinion (uniformly between min_num_seeds and max_num_seeds)
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
