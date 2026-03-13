import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    # For running the notebook
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuous-time Noisy Voter Model (CNVM)

    This package provides an efficient implementation of the CNVM, which is a dynamical system on a network of $N$ nodes (agents).
    Each node is endowed with one of $M$ discrete opinions. Thus, the system state is given by a vector $x \in \{1,\dots,M\}^N$, where $x_i$ describes the opinion of node $i$.
    Each node's opinion $x_i \in \{1,\dots,M\}$ changes over time according to a continuous-time Markov chain (Markov jump process).
    Given the current system state $x$, the generator matrix $Q^i$ of the continuous-time Markov chain associated with node $i$ is defined as

    $$Q^i \in \mathbb{R}^{M \times M},\quad (Q^i)_{m,n} := r_{m,n} \frac{d_{i,n}(x)}{(d_i)^\alpha} + \tilde{r}_{m,n},\ m\neq n,$$

    where $d_{i,n}(x)$ denotes the number of neighbors of node $i$ with opinion $n$ and $d_i$ is the degree of node $i$. The matrices $r, \tilde{r} \in \mathbb{R}^{M \times M}$ and $\alpha \in \mathbb{R}$ are model parameters.

    Thus, the transition rates $(Q^i)_{m,n}$ consist of two components. The first component $r_{m,n} \frac{d_{i,n}(x)}{(d_i)^\alpha}$ describes at which rate node $i$ gets "infected" by nodes in its neighborhood.
    The second part $\tilde{r}_{m,n}$ describes transitions that are independent from the neighborhood.

    It should be noted that after a node switches its opinion due to the above dynamics, the system state $x$ changes and hence all the generator matrices $Q^i$ may change as well.
    The model is simulated using a Gillespie-like algorithm that generates statistically correct samples.

    Let us conduct an example simulation of the CNVM. We begin by doing the necessary imports:
    """)
    return


@app.cell
def _():
    import networkx as nx
    import numpy as np
    from matplotlib import pyplot as plt

    return np, nx, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this example, we will have $N=100$ agents and $M=3$ different opinions. The rates $r_{m,n}$ and $\tilde{r}_{m,n}$ have to be specified for each pair $m\neq n$ and are thus defined as $M \times M$ arrays.
    (The diagonal entries are of no relevance.)
    It is also possible to set `r` or `r_tilde` to a single number, which then will be used for all $r_{m,n}$ or $\tilde{r}_{m,n}$ respectively.
    """)
    return


@app.cell
def _():
    num_agents = 100
    num_opinions = 3

    r = [[0, 0.8, 0.2], [0.2, 0, 0.9], [0.8, 0.3, 0]]
    r_tilde = 0.01
    return num_agents, num_opinions, r, r_tilde


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we specify the network structure using the `networkx` package. In this example, we sample a random graph using the Erdös-Renyi model.
    It should be noted that graphs with isolated vertices are not allowed in the CNVM model.
    Instead of a `networkx` graph we could also provide a neighbor list directly (i.e., a list where the i-th entry contains an array of the neighbors of node i).
    """)
    return


@app.cell
def _(num_agents, nx, plt):
    network = nx.erdos_renyi_graph(num_agents, p=0.1)
    nx.draw(network)
    plt.show()
    return (network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We summarize the model parameters in the class `CNVMParameters`.

    Providing the rates `r` and `r_tilde` is required to create an instance of `CNVMParameters`. Moreover, it is required to either specify a `network` (which we do), or a `network_generator` (from `sponet.network_generators`), or `num_agents` (in which case a complete network is used).
    The parameter `alpha` is optional with default value 1.

    Then we construct the model using the `CNVM` class and the parameters that we have defined before.
    """)
    return


@app.cell
def _(network, r, r_tilde):
    from sponet import CNVMParameters, CNVM

    params = CNVMParameters(
        network=network, r=r, r_tilde=r_tilde, alpha=1
    )
    model = CNVM(params)
    return model, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We execute the simulation by calling the method `CNVM.simulate`. Note that the first call of `CNVM.simulate` takes longer than subsequent calls because the simulation code is just-in-time compiled using `numba`.
    """)
    return


@app.cell
def _(model, num_agents):
    from sponet import sample_states_target_shares

    # sample a random initial state where 20% of nodes have opinion 0,
    # 50% have opinion 1, and 30% have opinion 2
    x_init = sample_states_target_shares(num_agents, [0.2, 0.5, 0.3])

    # simulation from t=0 to t=t_max
    t_max = 100

    # execute simulation
    t, x = model.simulate(t_max, x_init)
    print(f"t.shape = {t.shape}")  # time stamps of jumps
    print(f"x.shape = {x.shape}")  # system state after each jump
    return t, t_max, x, x_init


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to visualize the results of the simulation, we calculate an aggregate state called the *opinion shares*. The opinion shares are simply the numbers of agents of each discrete opinion. For example, in our initial state we had 20 agents with opinion 0, 50 with opinion 1, and 30 with opinion 2.

    The option `normalize=True` has the effect that instead of the absolute counts of each opinion we compute the normalized shares.
    """)
    return


@app.cell
def _(num_opinions, x):
    from sponet import OpinionShares

    opinion_shares = OpinionShares(num_opinions, normalize=True)
    c = opinion_shares(x)
    print(f"c.shape = {c.shape}")
    return c, opinion_shares


@app.cell
def _(c, mo, plt, t):
    def plot_traj():
        fig, ax = plt.subplots()
        for i in range(c.shape[1]):
            ax.plot(t, c[:, i], label=f"$c_{i + 1}$")
        ax.grid()
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("shares")
        return fig

    mo.mpl.interactive(plot_traj())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is often required to perform many simulations for different initial states to approximate important statistics. We provide the utility function `sample_many_runs` for that purpose.
    It would be hard to compare the trajectories if they would be stored at their different jump times.
    Hence, we have to provide the argument `t_eval` which is either an array of time points at which all the trajectories will be stored, or a number in which case the snapshots are stored equidistantly.
    """)
    return


@app.cell
def _(opinion_shares, params, t_max, x_init):
    from sponet import sample_many_runs

    t_many, c_many = sample_many_runs(
        params,
        x_init,
        t_max,
        t_eval=200,
        num_runs=5000,
        collective_variable=opinion_shares,
        n_jobs=-1,
    )
    print(f"t_many.shape = {t_many.shape}")
    print(f"c_many.shape = {c_many.shape}")
    return c_many, t_many


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output `c_many` has the shape (# samples, # timesteps, # opinions). Multiprocessing is enabled via n_jobs=-1 and disabled via n_jobs=None.

    We plot the ensemble average:
    """)
    return


@app.cell
def _(c_many, mo, np, plt, t_many):
    def plot_mean():
        fig, ax = plt.subplots()
        for i in range(c_many.shape[2]):
            ax.plot(t_many, np.mean(c_many[:, :, i], axis=0), label=f"$c_{i + 1}$")
        ax.grid()
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("counts")
        return fig

    mo.mpl.interactive(plot_mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `sponet` package also provides some approximations for the CNVM.

    For example, it is known that for certain networks, e.g., complete networks and dense Erdös-Rényi networks, the dynamics of the CNVM converges to a mean-field limit given by an ordinary differential equation (ODE) of the opinion shares.
    This ODE is also called mean-field equation or reaction-rate equation (RRE) and is defined as

    $$ \frac{d}{dt} c(t) = \sum_{\substack{m,n=1 \\ m\neq n}}^M c_m(t) (r_{mn} c_n(t) + \tilde{r}_{mn})(e_n - e_m), \quad \text{(RRE)} $$

    where $c(t)=(c_1(t), \dots, c_M(t))$ is the vector of opinion shares (e.g., $c_1(t) \in [0,1]$ is the share of nodes in state $1$), and $e_n$ is the $n$-th unit vector.
    (The proof can be found for example in [[Lücke et al., 2022]](https://arxiv.org/abs/2210.02934).)

    The RRE is accessible via the function `calc_rre_traj` that calculates the solution for us:
    """)
    return


@app.cell
def _(params, t_many, t_max):
    from sponet import calc_rre_traj

    t_rre, c_rre = calc_rre_traj(params, [0.2, 0.5, 0.3], t_max, t_eval=t_many)
    print(f"t_rre.shape = {t_rre.shape}")
    print(f"c_rre.shape = {c_rre.shape}")
    return c_rre, t_rre


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see below, the RRE does not agree particularly well with our model simulations.
    The reason is that the network is very small (100 nodes).
    Increasing the network size would result in a better match.

    (For smaller populations, the _chemical Langevin equation_ provides a much better approximation, see the `sample_cle` function in `sponet`.)
    """)
    return


@app.cell
def _(c_many, c_rre, mo, np, plt, t_many, t_rre):
    def plot_mean_rre():
        fig, ax = plt.subplots()
        ax.plot(t_many, np.mean(c_many[:, :, 0], axis=0), label=f"CNVM")
        ax.plot(t_rre, c_rre[:, 0], label=f"RRE")
        ax.grid()
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("$c_1$")
        return fig

    mo.mpl.interactive(plot_mean_rre())
    return


if __name__ == "__main__":
    app.run()
