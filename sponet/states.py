import numpy as np
from scipy.stats import dirichlet


def create_uniformly_random_states(
    num_agents: int, num_opinions: int, num_states: int
) -> np.ndarray:
    """
    Sample uniformly random states.

    The states are such that the shares of each opinion are uniform
    on the simplex of opinion shares.
    In addition, each state is randomly shuffled.

    Parameters
    ----------
    num_agents : int
    num_opinions : int
    num_states : int

    Returns
    -------
    np.ndarray
    """
    x = np.zeros((num_states, num_agents))
    alpha = np.ones(num_opinions)
    for i in range(num_states):
        shares = dirichlet.rvs(alpha=alpha)[0]
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
                create_uniformly_random_states(num_agents, num_opinions, missing_points),
            ]
        )
        x = np.unique(x.astype(int), axis=0)

    return x
