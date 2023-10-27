import time

import matplotlib.pyplot as plt
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
    idx = table[np.random.randint(0, len(table))]
    if idx > -1:
        return idx

    return np.searchsorted(spillover_cumsum, np.random.random(), side="right")


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
    print(len(table))
    table.extend([-1] * (factor - len(table)))

    spill_over = probabilities * factor - np.floor(probabilities * factor)
    spill_over /= np.sum(spill_over)

    return np.array(table), np.cumsum(spill_over)


def benchmark_sampling():
    n_list = [100, 1000, 10000, 100000]
    timeout = 5

    performance_cdf = []
    performance_table = []

    for n in n_list:
        print(n)
        expected_value = n / 10
        param_geom = 1 / expected_value
        probability_vector = [(1 - param_geom) ** k * param_geom for k in range(n)]
        probability_vector = np.array(probability_vector) / np.sum(probability_vector)

        # cdf sampling
        cum_sum = np.cumsum(probability_vector)
        rand_index_numba(cum_sum)  # compile
        num_iter = 0
        start = time.time()
        while time.time() < start + timeout:
            rand_index_numba(cum_sum)
            num_iter += 1
        end = time.time()
        performance_cdf.append((end - start) / num_iter)

        # table sampling
        table, spillover_cumsum = construct_table(probability_vector)
        rand_index_table(table, spillover_cumsum)  # compile
        num_iter = 0
        start = time.time()
        while time.time() < start + timeout:
            rand_index_table(table, spillover_cumsum)
            num_iter += 1
        end = time.time()
        performance_table.append((end - start) / num_iter)

    plt.loglog(n_list, performance_cdf, label="cdf")
    plt.loglog(n_list, performance_table, label="table")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    benchmark_sampling()
    # p = np.array([0.4581, 0.0032, 0.1985, 0.3298, 0.0022, 0.0080, 0.0002])
    #
    # table_p, spillover_cum_sum = construct_table(p)
    # results = []
    # for i in range(100):
    #     results.append(rand_index_table(table_p, spillover_cum_sum))
    #
    # unique, counts = np.unique(results, return_counts=True)
    # p_sample = counts / len(results)
    # print(p_sample)
