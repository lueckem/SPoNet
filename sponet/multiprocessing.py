import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from tqdm import tqdm

from .cntm.model import CNTM
from .cntm.parameters import CNTMParameters
from .cnvm.model import CNVM
from .cnvm.parameters import CNVMParameters
from .collective_variables import CollectiveVariable
from .parameters import Parameters


def sample_many_runs(
    params: Parameters,
    initial_states: NDArray,
    t_max: float,
    num_timesteps: int,
    num_runs: int,
    n_jobs: int | None = None,
    collective_variable: CollectiveVariable | None = None,
    rng: Generator = default_rng(),
    progress_bar: bool = False,
) -> tuple[NDArray, NDArray]:
    """
    Sample multiple runs of the model specified by params.

    Parameters
    ----------
    params : Parameters
        Either CNVM or CNTM Parameters.
        If a NetworkGenerator is used, a new network will be sampled for every run.
    initial_states : NDArray
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
    rng : Generator, optional
        Random number generator.
    progress_bar : bool, optional
        Whether to print the progress.
        If multiprocessing is used, only the progress of the first subprocess is printed.

    Returns
    -------
    t_out, x_out : tuple[NDArray, NDArray]
        (t_out, x_out), time_out.shape = (num_timesteps,),
        x_out.shape = (num_initial_states, num_runs, num_timesteps, num_agents)
    """
    t_out = np.linspace(0, t_max, num_timesteps)
    worker = _Worker(params, t_max, num_timesteps, collective_variable)

    # no multiprocessing
    if n_jobs is None or n_jobs == 1:
        x_out = worker(
            initial_states,
            num_runs,
            rng,
            progress_bar,
        )
        return t_out, x_out

    # multiprocessing
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    rngs = rng.spawn(n_jobs)

    progress_bars = [False] * n_jobs
    if progress_bar:
        progress_bars[0] = True

    if num_runs >= initial_states.shape[0]:  # parallelization along runs
        states_chunks = [initial_states] * n_jobs
        runs_chunks = _split_runs(num_runs, n_jobs)
        concat_axis = 1
    else:  # parallelization along initial states
        states_chunks = np.array_split(initial_states, n_jobs)
        runs_chunks = [num_runs] * n_jobs
        concat_axis = 0

    with ProcessPoolExecutor() as executor:
        x_out = list(
            executor.map(worker, states_chunks, runs_chunks, rngs, progress_bars)
        )
    x_out = np.concatenate(x_out, axis=concat_axis)
    return t_out, x_out


class _Worker:
    def __init__(
        self,
        params: Parameters,
        t_max: float,
        num_timesteps: int,
        collective_variable: CollectiveVariable | None,
    ):
        self.params = params
        self.t_max = t_max
        self.num_timesteps = num_timesteps
        self.cv = collective_variable
        self.num_opinions = params.num_opinions
        self.num_agents = params.num_agents

    def __call__(
        self,
        initial_states: NDArray,
        num_runs: int,
        rng: Generator,
        progress_bar: bool = False,
    ) -> NDArray:
        num_initial_states = initial_states.shape[0]

        if isinstance(self.params, CNVMParameters):
            model = CNVM(self.params)
        elif isinstance(self.params, CNTMParameters):
            model = CNTM(self.params)
        else:
            raise ValueError("Parameters not valid.")

        if self.cv is None:
            opinion_dtype = np.min_scalar_type(self.num_opinions - 1)
            x_out = np.zeros(
                (num_initial_states, num_runs, self.num_timesteps, self.num_agents),
                dtype=opinion_dtype,
            )
        else:
            x_out = np.zeros(
                (
                    num_initial_states,
                    num_runs,
                    self.num_timesteps,
                    self.cv.dimension,
                )
            )

        pbar = tqdm(total=num_initial_states * num_runs) if progress_bar else None

        for j in range(num_initial_states):
            for i in range(num_runs):
                _, x = model.simulate(
                    self.t_max,
                    len_output=self.num_timesteps,
                    x_init=initial_states[j],
                    rng=rng,
                )
                if self.cv is None:
                    x_out[j, i, :, :] = x
                else:
                    x_out[j, i, :, :] = self.cv(x)

                if pbar is not None:
                    pbar.update()

        return x_out


def _split_runs(num_runs: int, num_chunks: int) -> np.ndarray:
    """
    Split num_runs into num_chunks approximately equal chunks.
    """
    chunks = np.ones(num_chunks, dtype=int) * (num_runs // num_chunks)
    chunks[: (num_runs % num_chunks)] += 1
    return chunks
