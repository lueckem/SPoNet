import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator, default_rng
from numba import njit

# todo: const complexity is not needed anymore


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
    table_prob = weights / np.sum(weights) * weights.shape[0]
    table_alias = np.zeros(weights.shape[0], dtype=np.int32)

    small_ids = [i for i in range(table_prob.shape[0]) if table_prob[i] < 1]
    large_ids = [i for i in range(table_prob.shape[0]) if table_prob[i] >= 1]

    while small_ids and large_ids:
        small_id = small_ids.pop()
        large_id = large_ids.pop()

        table_alias[small_id] = large_id
        table_prob[large_id] = table_prob[large_id] + table_prob[small_id] - 1

        small_ids.append(large_id) if table_prob[large_id] < 1 else large_ids.append(large_id)

    while large_ids:
        table_prob[large_ids.pop()] = 1

    while small_ids:
        table_prob[small_ids.pop()] = 1

    return table_prob, table_alias


@njit()
def sample_from_alias(table_prob: np.ndarray, table_alias: np.ndarray, rng: Generator) -> int:
    u = rng.random()
    idx = int(u * table_prob.shape[0])
    y = table_prob.shape[0] * u - idx
    if y < table_prob[idx]:
        return idx
    return table_alias[idx]


@njit()
def bench_alias(n: int, table_prob: np.ndarray, table_alias: np.ndarray, rng: Generator) -> np.ndarray:
    results = np.zeros(n)
    for i in range(n):
        results[i] = sample_from_alias(table_prob, table_alias, rng)
    return results


@njit()
def bench_bisect(n: int, cumsum: np.ndarray, rng: Generator) -> np.ndarray:
    results = np.zeros(n)
    for i in range(n):
        results[i] = sample_weighted_bisect(cumsum, rng)
    return results


@njit()
def bench_uniform(n: int, high: int,  rng: Generator) -> np.ndarray:
    results = np.zeros(n)
    for i in range(n):
        results[i] = sample_randint(high, rng)
    return results


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


def benchmark_sampling():
    n_list = [100, 1000, 10000, 100000, 1000000]
    num_iter = 1000000

    performance_cdf = []
    performance_table = []
    performance_uniform = []

    rng = default_rng()

    for n in n_list:
        print(n)
        expected_value = n / 10
        param_geom = 1 / expected_value
        probability_vector = [(1 - param_geom) ** k * param_geom for k in range(n)]
        probability_vector = np.array(probability_vector) / np.sum(probability_vector)
        probability_vector = np.ones(n) / n

        # cdf sampling
        cum_sum = np.cumsum(probability_vector)
        bench_bisect(1, cum_sum, rng)  # compile
        start = time.time()
        bench_bisect(num_iter, cum_sum, rng)
        end = time.time()
        performance_cdf.append(end - start)

        # table sampling
        tp, ta = build_alias_table(probability_vector)
        bench_alias(1, tp, ta, rng)  # compile
        start = time.time()
        bench_alias(num_iter, tp, ta, rng)
        end = time.time()
        performance_table.append(end - start)

        # uniform sampling
        bench_uniform(1, n, rng)  # compile
        start = time.time()
        bench_uniform(num_iter, n, rng)
        end = time.time()
        performance_uniform.append(end - start)

    plt.loglog(n_list, performance_cdf, label="cdf")
    plt.loglog(n_list, performance_table, label="table")
    plt.loglog(n_list, performance_uniform, label="uniform")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    benchmark_sampling()
    # rng = default_rng()
    #
    # p = np.array([0.4581, 0.0032, 0.1985, 0.3298, 0.0022, 0.0080, 0.0002])
    # tp, ta = build_alias_table(p)
    # sample_alias(tp, ta, rng)
    # # print(tp)
    # # print(ta)
    #
    # results = []
    # start = time.time()
    # for i in range(10000000):
    #     results.append(sample_alias(tp, ta, rng))
    # end = time.time()
    #
    # unique, counts = np.unique(results, return_counts=True)
    # p_sample = counts / len(results)
    # print(p_sample)
    # print(end - start)

