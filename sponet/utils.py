import numpy as np
from numpy.random import Generator
from numba import njit
from scipy.integrate import solve_ivp
import multiprocessing as mp
from matplotlib.axes import Axes

from .collective_variables import CollectiveVariable
from .parameters import Parameters

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
    rng: Generator = np.random.default_rng(),
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
    rng : Generator, optional
        random number generator

    Returns
    -------
    t_out, x_out : tuple[np.ndarray, np.ndarray]
        (t_out, x_out), time_out.shape = (num_timesteps,),
        x_out.shape = (num_initial_states, num_runs, num_timesteps, num_agents)
    """
    t_out = np.linspace(0, t_max, num_timesteps)

    # no multiprocessing
    if n_jobs is None or n_jobs == 1:
        x_out = _sample_many_runs_subprocess(
            params,
            initial_states,
            t_max,
            num_timesteps,
            num_runs,
            rng,
            collective_variable,
        )
        return t_out, x_out

    # multiprocessing
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    if num_runs >= initial_states.shape[0]:  # parallelization along runs
        chunks = _split_runs(num_runs, n_jobs)
        processes = [
            (
                params,
                initial_states,
                t_max,
                num_timesteps,
                chunk,
                rng,
                collective_variable,
            )
            for chunk in chunks
        ]
        concat_axis = 1

    else:  # parallelization along initial states
        chunks = np.array_split(initial_states, n_jobs)
        processes = [
            (
                params,
                chunk,
                t_max,
                num_timesteps,
                num_runs,
                rng,
                collective_variable,
            )
            for chunk in chunks
        ]
        concat_axis = 0

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
    rng: Generator,
    collective_variable: CollectiveVariable = None,
) -> np.ndarray:
    t_out = np.linspace(0, t_max, num_timesteps)
    num_initial_states = initial_states.shape[0]

    if isinstance(params, CNVMParameters):
        model_type = CNVM
    elif isinstance(params, CNTMParameters):
        model_type = CNTM
    else:
        raise ValueError("Parameters not valid.")
    model = model_type(params)

    if collective_variable is None:
        x_out = np.zeros(
            (num_initial_states, num_runs, num_timesteps, model.params.num_agents)
        )
    else:
        x_out = np.zeros(
            (num_initial_states, num_runs, num_timesteps, collective_variable.dimension)
        )

    for j in range(num_initial_states):
        for i in range(num_runs):
            t, x = model.simulate(
                t_max, len_output=4 * num_timesteps, x_init=initial_states[j], rng=rng
            )
            t_ind = argmatch(t_out, t)
            if collective_variable is None:
                x_out[j, i, :, :] = x[t_ind, :]
            else:
                x_out[j, i, :, :] = collective_variable(x[t_ind, :])
    return x_out


def _split_runs(num_runs: int, num_chunks: int) -> np.ndarray:
    """
    Split num_runs into num_chunks approximately equal chunks.
    """
    chunks = np.ones(num_chunks, dtype=int) * (num_runs // num_chunks)
    chunks[: (num_runs % num_chunks)] += 1
    return chunks


def calc_rre_traj(
    params: CNVMParameters, c_0: np.ndarray, t_max: float, t_eval=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the RRE given by parameters, starting from c_0, up to time t_max.

    Solves the ODE using scipy's "solve_ivp".

    Parameters
    ----------
    params : CNVMParameters
    c_0 : np.ndarray
        Initial state, shape=(num_opinions,)
    t_max : float
        End time.
    t_eval : np.ndarray, optional
        Time points, at which the solution should be evaluated.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        1. timepoints, shape=(?,).
        2. c, shape=(?, num_opinions).
    """

    def rhs(t, c):
        out = np.zeros_like(c)
        for m in range(params.num_opinions):
            for n in range(params.num_opinions):
                if n == m:
                    continue

                state_change = np.zeros_like(c)
                state_change[m] = -1
                state_change[n] = 1

                prop = c[m] * (
                    params.r_imit * params.prob_imit[m, n] * c[n]
                    + params.r_noise * params.prob_noise[m, n] / params.num_opinions
                )

                out += prop * state_change
        return out

    sol = solve_ivp(rhs, (0, t_max), c_0, rtol=1e-8, atol=1e-8, t_eval=t_eval)
    return sol.t, sol.y.T


def plot_trajectories(
    params: Parameters,
    t: float,
    cv: CollectiveVariable,
    ax: Axes,
    num_initial_states: int = 3,
    samples_per_state: int = 5,
) -> Axes:
    """
    Plot some trajectories with uniformly random initial states
    in the provided Axes object.

    The purpose of this function is to quickly inspect how trajectories
    look like for the given set of Parameters.

    Parameters
    ----------
    params : Parameters
    t : float
    cv : CollectiveVariable
    ax : Axes
    num_initial_states : int, optional
    samples_per_state : int, optional

    Returns
    -------
    Axes
    """
    initial_states = np.random.randint(
        0, params.num_opinions, size=(num_initial_states, params.num_agents)
    )

    t, c = sample_many_runs(
        params,
        initial_states,
        t,
        1000,
        samples_per_state,
        collective_variable=cv,
    )

    colors = ["k", "b", "g", "c", "r", "y"]
    linestyles = ["-", "--", "-."]

    for i in range(num_initial_states):
        this_linestyle = linestyles[i % len(linestyles)]
        for j in range(samples_per_state):
            this_color = colors[j % len(colors)]
            ax.plot(t, c[i, j, :, 0], linestyle=this_linestyle, color=this_color)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$c_0$")
    ax.grid()

    return ax


@njit
def argmatch(x_ref, x):
    """
    Find indices such that |x[indices] - x_ref| = min!

    Parameters
    ----------
    x_ref : np.ndarray
        1D, sorted
    x : np.ndarray
        1D, sorted
    Returns
    -------
    np.ndarray
    """
    size = np.shape(x_ref)[0]
    out = np.zeros(size, dtype=np.int64)
    ref_ind = 0
    ind = 0
    while x_ref[ref_ind] < x[ind]:
        ref_ind += 1

    while ref_ind < size and ind < x.shape[0] - 1:
        if x[ind] <= x_ref[ref_ind] <= x[ind + 1]:
            if np.abs(x[ind] - x_ref[ref_ind]) < np.abs(
                x[ind + 1] - x_ref[ref_ind]
            ):  # smaller is nearer
                out[ref_ind] = ind
            else:  # bigger is nearer
                out[ref_ind] = ind + 1
            ref_ind += 1
        else:
            ind += 1

    while ref_ind < size:
        out[ref_ind] = x.shape[0] - 1
        ref_ind += 1

    return out
