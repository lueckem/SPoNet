import numpy as np
from numpy.random import Generator
from numba import njit


@njit()
def sample_randint(high_excl: int, rng: Generator) -> int:
    """
    Sample uniformly random integer in {0, ..., high_excl - 1}.

    It is faster than numpy.randint for single samples.
    It has a detectable bias if high_excl is really large. (For high_excl < 10^10 it is fine though.)

    Parameters
    ----------
    high_excl : int
    rng : Generator
        random number generator

    Returns
    -------
    int
    """
    return int(rng.random() * high_excl)


@njit()
def build_alias_table(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the probability table and alias table for given weights.

    The tables can be used for an O(1) sampling of the weighted discrete distribution
    with Prob(i) = weights[i] / sum(weights).

    Parameters
    ----------
    weights : np.ndarray

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        probability table, alias table
    """
    table_prob = weights / np.sum(weights) * weights.shape[0]
    table_alias = np.zeros(weights.shape[0], dtype=np.int32)

    small_ids = [i for i in range(table_prob.shape[0]) if table_prob[i] < 1]
    large_ids = [i for i in range(table_prob.shape[0]) if table_prob[i] >= 1]

    while small_ids and large_ids:
        small_id = small_ids.pop()
        large_id = large_ids.pop()

        table_alias[small_id] = large_id
        table_prob[large_id] = table_prob[large_id] + table_prob[small_id] - 1

        (
            small_ids.append(large_id)
            if table_prob[large_id] < 1
            else large_ids.append(large_id)
        )

    while large_ids:
        table_prob[large_ids.pop()] = 1

    while small_ids:
        table_prob[small_ids.pop()] = 1

    return table_prob, table_alias


@njit()
def sample_from_alias(
    table_prob: np.ndarray, table_alias: np.ndarray, rng: Generator
) -> int:
    """
    Sample from weighted discrete distribution given by the probability and alias tables.

    Sampling has O(1) complexity.
    Should only be used if len(table_prob) < 10^10 due to bias for large numbers.

    Parameters
    ----------
    table_prob : np.ndarray
    table_alias : np.ndarray
    rng : Generator

    Returns
    -------
    int
    """
    u = rng.random()
    idx = int(u * table_prob.shape[0])
    y = table_prob.shape[0] * u - idx
    if y < table_prob[idx]:
        return idx
    return table_alias[idx]


@njit()
def sample_weighted_bisect(prob_cum_sum: np.ndarray, rng: Generator) -> int:
    """
    Sample random index 0 <= i < len(prob_cumsum) according to probability distribution.

    Via bisection of CDF (log complexity).

    Parameters
    ----------
    prob_cum_sum : np.ndarray
        1D array containing the cumulative probabilities, i.e.,
        the first entry is the probability of choosing index 0,
        the second entry the probability of choosing index 0 or 1, and so on.
        The last entry is 1.
    rng : Generator
        random number generator

    Returns
    -------
    int
    """
    return np.searchsorted(prob_cum_sum, rng.random(), side="right")
