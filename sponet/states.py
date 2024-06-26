import numpy as np
from numpy.random import Generator, default_rng


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
        np.random.shuffle(this_x)
        x[i] = this_x

    x = np.unique(x.astype(int), axis=0)

    while x.shape[0] != num_states:
        missing_points = num_states - x.shape[0]
        x = np.concatenate(
            [
                x,
                sample_states_uniform_shares(num_agents, num_opinions, missing_points),
            ]
        )
        x = np.unique(x.astype(int), axis=0)

    return x
