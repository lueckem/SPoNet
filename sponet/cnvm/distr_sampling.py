import numpy as np
from numba import njit


@njit()
def rand_index_numba(prob_cum_sum) -> int:
    """
    Sample random index 0 <= i < len(prob_cumsum) according to probability distribution.

    Parameters
    ----------
    prob_cum_sum : np.ndarray
        1D array containing the cumulative probabilities, i.e.,
        the first entry is the probability of choosing index 0,
        the second entry the probability of choosing index 0 or 1, and so on.
        The last entry is 1.

    Returns
    -------
    int
    """
    return np.searchsorted(prob_cum_sum, np.random.random(), side="right")


@njit()
def rand_index_table(table, spillover_cumsum) -> int:
    idx = np.random.choice(table)
    if idx > -1:
        return idx

    return rand_index_numba(spillover_cumsum)


@njit()
def construct_table(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    factor = 16
    prob_spill_over = 1
    numbers_of_cells = None

    while prob_spill_over > 0.01:
        factor *= 2
        numbers_of_cells = np.floor(probabilities * factor).astype(np.int32)
        prob_spill_over = 1 - np.sum(numbers_of_cells) / factor

    table = []
    for i in range(probabilities.shape[0]):
        table.extend([i] * numbers_of_cells[i])
    table.extend([-1] * (factor - len(table)))

    spill_over = probabilities * factor - np.floor(probabilities * factor)
    spill_over /= np.sum(spill_over)

    return np.array(table), np.cumsum(spill_over)


def benchmark_sampling():
    n_list = [10, 100, 1000, 10000, 100000]

    for n in n_list:
        p = np.random.random(n)
        p /= np.sum(p)


if __name__ == '__main__':
    p = np.array([0.4581, 0.0032, 0.1985, 0.3298, 0.0022, 0.0080, 0.0002])
    p = np.random.random(10000)
    p /= np.sum(p)

    table_p, spillover_cum_sum = construct_table(p)
    results = []
    for i in range(100):
        results.append(rand_index_table(table_p, spillover_cum_sum))

    unique, counts = np.unique(results, return_counts=True)
    p_sample = counts / len(results)
    print(p_sample)
