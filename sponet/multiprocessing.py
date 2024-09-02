import numpy as np
import multiprocessing as mp
import time
from datetime import timedelta

from .collective_variables import CollectiveVariable
from .parameters import Parameters
from .utils import argmatch

from .cnvm.parameters import CNVMParameters
from .cnvm.model import CNVM

from .cntm.parameters import CNTMParameters
from .cntm.model import CNTM


def sample_many_runs(
    params: Parameters,
    initial_states: np.ndarray,
    t_max: float,
    num_timesteps: int,
    num_runs: int,
    n_jobs: int = None,
    collective_variable: CollectiveVariable = None,
    seed: int = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample multiple runs of the model specified by params.

    Parameters
    ----------
    params : Parameters
        Either CNVM or CNTM Parameters.
        If a NetworkGenerator is used, a new network will be sampled for every run.
    initial_states : np.ndarray
        Array of initial states, shape = (num_initial_states, num_agents).
        Num_runs simulations will be executed for each initial state.
    t_max : float
        End time.
    num_timesteps : int
        Trajectory will be saved at equidistant time points np.linspace(0, t_max, num_timesteps).
    num_runs : int
        Number of samples.
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
        If multiprocessing is used, only the progress of the first subprocess is printed.


    Returns
    -------
    t_out, x_out : tuple[np.ndarray, np.ndarray]
        (t_out, x_out), time_out.shape = (num_timesteps,),
        x_out.shape = (num_initial_states, num_runs, num_timesteps, num_agents)
    """
    t_out = np.linspace(0, t_max, num_timesteps)

    if seed is None:
        seed = np.random.default_rng().integers(2**31)

    # no multiprocessing
    if n_jobs is None or n_jobs == 1:
        x_out = _sample_many_runs_subprocess(
            params,
            initial_states,
            t_max,
            num_timesteps,
            num_runs,
            seed,
            verbose,
            collective_variable,
        )
        return t_out, x_out

    # multiprocessing
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    if num_runs >= initial_states.shape[0]:  # parallelization along runs
        chunks = _split_runs(num_runs, n_jobs)
        processes = [
            [
                params,
                initial_states,
                t_max,
                num_timesteps,
                chunk,
                seed + i,
                False,
                collective_variable,
            ]
            for i, chunk in enumerate(chunks)
        ]
        concat_axis = 1

    else:  # parallelization along initial states
        chunks = np.array_split(initial_states, n_jobs)
        processes = [
            [
                params,
                chunk,
                t_max,
                num_timesteps,
                num_runs,
                seed + i,
                False,
                collective_variable,
            ]
            for i, chunk in enumerate(chunks)
        ]
        concat_axis = 0

    if verbose:
        processes[0][6] = True

    with mp.Pool(n_jobs) as pool:
        x_out = pool.starmap(_sample_many_runs_subprocess, processes)
    x_out = np.concatenate(x_out, axis=concat_axis)

    return t_out, x_out


def _sample_many_runs_subprocess(
    params: Parameters,
    initial_states: np.ndarray,
    t_max: float,
    num_timesteps: int,
    num_runs: int,
    seed: int,
    verbose: bool,
    collective_variable: CollectiveVariable = None,
) -> np.ndarray:
    t_out = np.linspace(0, t_max, num_timesteps)
    num_initial_states = initial_states.shape[0]
    rng = np.random.default_rng(seed)

    if isinstance(params, CNVMParameters):
        model_type = CNVM
    elif isinstance(params, CNTMParameters):
        model_type = CNTM
    else:
        raise ValueError("Parameters not valid.")
    model = model_type(params)

    if collective_variable is None:
        opinion_dtype = np.min_scalar_type(params.num_opinions - 1)

        x_out = np.zeros(
            (num_initial_states, num_runs, num_timesteps, model.params.num_agents),
            dtype=opinion_dtype,
        )
    else:
        x_out = np.zeros(
            (num_initial_states, num_runs, num_timesteps, collective_variable.dimension)
        )

    num_iter = 0
    total_num_iter = num_initial_states * num_runs
    iter_delta = round(total_num_iter / 20)
    next_print_iter = iter_delta
    start_time = time.time()
    if verbose:
        print("t=0:00:00 : 0%.")

    for j in range(num_initial_states):
        for i in range(num_runs):
            num_iter += 1
            t, x = model.simulate(
                t_max, len_output=4 * num_timesteps, x_init=initial_states[j], rng=rng
            )
            t_ind = argmatch(t_out, t)
            if collective_variable is None:
                x_out[j, i, :, :] = x[t_ind, :]
            else:
                x_out[j, i, :, :] = collective_variable(x[t_ind, :])

            if verbose and num_iter >= next_print_iter:
                elapsed_time = time.time() - start_time
                estimated_duration = elapsed_time / (num_iter / total_num_iter)
                estimated_time_left = timedelta(
                    seconds=round(estimated_duration - elapsed_time)
                )
                elapsed_time = timedelta(seconds=round(elapsed_time))
                percentage = round(num_iter / total_num_iter * 100)
                print(
                    f"t={elapsed_time} : {percentage}%. (Time remaining ~{estimated_time_left})"
                )
                next_print_iter += iter_delta
    return x_out


def _split_runs(num_runs: int, num_chunks: int) -> np.ndarray:
    """
    Split num_runs into num_chunks approximately equal chunks.
    """
    chunks = np.ones(num_chunks, dtype=int) * (num_runs // num_chunks)
    chunks[: (num_runs % num_chunks)] += 1
    return chunks
