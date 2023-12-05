import numpy as np
from matplotlib.axes import Axes
from .collective_variables import CollectiveVariable
from .parameters import Parameters
from .multiprocessing import sample_many_runs


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
