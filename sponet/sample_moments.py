import multiprocessing as mp
from typing import Optional

import numpy as np

from sponet.multiprocessing import sample_many_runs

from .collective_variables import CollectiveVariable
from .parameters import Parameters


def sample_moments(
    params: Parameters,
    initial_state: np.ndarray,
    t_max: float,
    num_timesteps: int,
    batch_size: int,
    rel_tol: float = 1e-3,
    n_jobs: Optional[int] = None,
    collective_variable: Optional[CollectiveVariable] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Sample in batches and estimate first and second moments.

    Estimates a confidence interval for the mean (first moment) and
    samples until the size of the confidence interval relative to the mean is less than a tolerance:
    [4 sigma / sqrt(N)] / mu < rel_tol.

    Parameters
    ----------
    params : Parameters
        Either CNVM or CNTM Parameters.
        If a NetworkGenerator is used, a new network will be sampled for every run.
    initial_states : np.ndarray
        Initial state, shape = (num_agents,).
    t_max : float
        End time.
    num_timesteps : int
        Trajectory will be saved at equidistant time points np.linspace(0, t_max, num_timesteps).
    batch_size : int
        Number of samples per batch.
    rel_tol : float, optional
        Stop when the relative size of the 95% confidence interval is smaller than the tolerance.
    n_jobs : int, optional
        If "None", no multiprocessing is applied. If "-1", all available CPUs will be used.
    collective_variable : CollectiveVariable, optional
        If collective variable is specified, the projected trajectory will be returned
        instead of the full trajectory.
    seed : int, optional
        Seed for random number generation.
        If multiprocessing is used, the subprocesses receive the seeds {seed, seed + 1, ...}.
    verbose : bool, optional
        Whether to print the progress.

    Returns
    -------
    t, mean, var, num_samples : tuple[np.ndarray, np.ndarray, np.ndarray, int]
        t.shape = (num_timesteps,),
        mean.shape = (num_timesteps, num_agents)
        var.shape = (num_timesteps, num_agents)
    """
    num_a = (
        params.num_agents
        if collective_variable is None
        else collective_variable.dimension
    )

    if seed is None:
        seed = np.random.default_rng().integers(2**31)

    if n_jobs is None:
        n_jobs = 1
    elif n_jobs == -1:
        n_jobs = mp.cpu_count()

    sum_x = np.zeros((num_timesteps, num_a))
    sum_xx = np.zeros((num_timesteps, num_a))
    num_samples = 0

    while True:
        print(f"Total number of samples: {num_samples}.")
        t, x = sample_many_runs(
            params,
            np.array([initial_state]),
            t_max,
            num_timesteps,
            batch_size,
            n_jobs=n_jobs,
            collective_variable=collective_variable,
            seed=seed,
        )
        seed += n_jobs
        num_samples += batch_size
        sum_x += np.sum(x[0], axis=0)
        sum_xx += np.sum(x[0] ** 2, axis=0)

        mean = sum_x / num_samples
        variance = sum_xx / num_samples - mean**2
        confidence_size = 4 * np.sqrt(variance / num_samples)
        rel_error = np.max(confidence_size / mean)
        print(f"Relative error: {rel_error}.\n")
        if rel_error < rel_tol:
            return t, mean, variance, num_samples
