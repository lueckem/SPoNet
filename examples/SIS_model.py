import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    # for running the notebook
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The SIS model and epidemic thresholds

    In this notebook we examine the *Susceptible-Infectious-Susceptible* (SIS) model of epidemiology.
    The SIS model describes how an infectious disease spreads on a contact network of people.
    Each node in this network represents a person and can be either susceptible (S) or infectious (I).
    If a susceptible node has an infectious neighbor, it becomes infectious itself at a certain rate (infection rate).
    An infectious node becomes susceptible again at the recovery rate, which we set to $1$.

    Hence, the SIS model is a special case of the CNVM. It is given by the rate parameters

    $$ r = \begin{pmatrix} - & \lambda \\ 0 & - \end{pmatrix}, \quad \tilde{r} = \begin{pmatrix} - & 0 \\ 1 & - \end{pmatrix}. $$

    Typically, the parameter $\alpha$ of the CNVM is set to $0$ in the SIS model, which means that the rate at which a susceptible node gets infected scales linearly with the number of infectious neighbors.
    However, for simplicity we will use the default $\alpha = 1$ in this notebook.

    The behavior of the SIS model heavily depends on the underlying network structure.
    We chose a random Erdös-Renyi graph in this example because it is well understood.

    Let us start by doing the necessary imports and defining the model.
    """)
    return


@app.cell
def _():
    import numpy as np
    import networkx as nx
    from matplotlib import pyplot as plt

    from sponet import CNVMParameters, CNVM, OpinionShares
    from sponet import sample_many_runs, calc_rre_traj

    return (
        CNVM,
        CNVMParameters,
        OpinionShares,
        calc_rre_traj,
        np,
        nx,
        plt,
        sample_many_runs,
    )


@app.cell
def _(OpinionShares, nx):
    # -- constants --
    num_opinions = 2
    num_agents = 1000
    network = nx.erdos_renyi_graph(num_agents, p=0.1)
    cv = OpinionShares(
        num_opinions, normalize=True
    )  # for measuring the percentage of infectious nodes
    return cv, network, num_agents


@app.cell
def _(CNVMParameters, network):
    # helper function to create parameters for varying infection rates
    def params_from_infection_rate(infection_rate: float) -> CNVMParameters:
        return CNVMParameters(
            network=network,
            r=[[0, infection_rate], [0, 0]],
            r_tilde=[[0, 0], [1, 0]]
            )

    return (params_from_infection_rate,)


@app.cell
def _(CNVM, params_from_infection_rate):
    # helper function to create models for varying infection rates
    def model_from_infection_rate(infection_rate: float) -> CNVM:
        return CNVM(params_from_infection_rate(infection_rate))

    return (model_from_infection_rate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we define the simulation parameters, let the model run, and plot the results.
    We always start with 30% infectious nodes and plot the evolution of the share of infectious nodes.

    Observe how the trajectories differ for varying infection rate $\lambda$ using the below slider.
    """)
    return


@app.cell
def _(np, num_agents):
    # -- simulation parameters --
    t_max = 100
    t_eval = 1001
    x_init = np.zeros(num_agents)  # initial state
    x_init[: int(num_agents * 0.3)] = 1
    np.random.shuffle(x_init)
    return t_eval, t_max, x_init


@app.cell
def _(cv, model_from_infection_rate, np, plt, t_eval, t_max, x_init):
    # plot a trajectory of the model depending on the infection rate `ir`
    def plot_traj(ir: float):
        # simulate model
        t, x = model_from_infection_rate(ir).simulate(
            t_max=t_max,
            x_init=x_init,
            t_eval=t_eval,
            rng=np.random.default_rng(123),
        )
        c = cv(x)  # calculate the share of infectious nodes

        # create plot
        fig, ax = plt.subplots()
        ax.plot(t, c[:, 1])
        ax.grid()
        ax.set_xlabel("t")
        ax.set_ylabel("percentage infected")
        ax.set_ylim(-0.03, 0.6)
        return fig

    return (plot_traj,)


@app.cell
def _(mo):
    infection_rate = mo.ui.slider(0.1, 2.0, 0.1, show_value=True, value=1.0)
    mo.md(f"Infection rate $\\lambda$ = {infection_rate}")
    return (infection_rate,)


@app.cell
def _(infection_rate, mo, plot_traj):
    mo.mpl.interactive(plot_traj(infection_rate.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above plot shows that the disease dies out quickly if the infection rate is rather small since the percentage of infected nodes goes to 0.
    If the infection rate is large, however, the disease does not die out.
    Instead, the share of infected nodes stabilizes around a value larger than 0.

    Let us conduct a statistical analysis of this behavior for different infection rates. In the following code block we perform many simulations of the SIS model. (If this takes too long on your machine, try reducing the number of samples by modifying the `num_runs` parameter.)
    """)
    return


@app.cell
def _(cv, params_from_infection_rate, sample_many_runs, t_eval, t_max, x_init):
    infection_rates = [0.8, 0.9, 1.0, 1.1, 1.2]

    c_results = []  # collect trajectories for each ir in a list
    for ir in infection_rates:
        params = params_from_infection_rate(ir)
        t, c = sample_many_runs(
            params=params,
            initial_states=x_init,
            t_max=t_max,
            t_eval=t_eval,
            num_runs=100,
            collective_variable=cv,  # return shares instead of full states
            n_jobs=-1,  # use multiprocessing
        )
        c_results.append(c)
    return c_results, infection_rates, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We plot the average share of infectious nodes and the probability that the disease survived up to time $t=100$ (i.e., there is at least one infectious node present).
    """)
    return


@app.cell
def _(c_results, infection_rates, mo, np, plt, t):
    def plot_results():
        fig, ax = plt.subplots()
        for ir, c_res in zip(infection_rates, c_results):
            ax.plot(t, np.mean(c_res[:, :, 1], axis=0), label=f"$\\lambda={ir}$")
        ax.legend()
        ax.grid()
        ax.set_xlabel("t")
        ax.set_ylabel("percentage infected")
        return fig

    mo.mpl.interactive(plot_results())
    return


@app.cell
def _(c_results, infection_rates, mo, np, plt):
    persistence_probabilities = [
        np.linalg.norm(c_r[:, -1, 1], 0) / c_r.shape[0] for c_r in c_results
    ]

    def plot_survival():
        fig, ax = plt.subplots()
        ax.plot(infection_rates, persistence_probabilities, "--x")
        ax.set_ylabel("infection survival probability")
        ax.set_xlabel("infection rate")
        ax.grid()
        return fig

    mo.mpl.interactive(plot_survival())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Apparently, the disease will likely die out if the infection rate is below 1, and will likely persist if the infection rate is above 1.
    This critical value is called the *epidemic threshold*.

    For this example (the SIS model on a sufficiently dense Erdös-Renyi random graph) it is known that the evolution of the share of infectious nodes $c(t)$ is given by the following mean-field ODE in the large population limit:

    $$ \frac{d}{dt} c(t) = -c(t) + \lambda (1 - c(t)) c(t). \qquad \text{(reaction-rate equation (RRE))} $$

    (See for example the paper [[Lücke et al., 2022]](https://arxiv.org/abs/2210.02934) for further information about the RRE.)

    The plot below shows that this ODE is already reasonably accurate for our finite size network.
    """)
    return


@app.cell
def _(
    c_results,
    calc_rre_traj,
    infection_rates,
    mo,
    np,
    params_from_infection_rate,
    plt,
    t,
    t_max,
):
    def plot_rre():
        idx = 2  # pick which infection rate
        ir = infection_rates[idx]
        params = params_from_infection_rate(ir)
        t_rre, c_rre = calc_rre_traj(params, [0.7, 0.3], t_max)
    
        fig, ax = plt.subplots()
        ax.plot(t, np.mean(c_results[idx][:, :, 1], axis=0), label="CNVM")
        ax.plot(t_rre, c_rre[:, 1], "--k", label="RRE")
        ax.legend()
        ax.grid()
        ax.set_xlabel("t")
        ax.set_ylabel("percentage infected")
        return fig

    mo.mpl.interactive(plot_rre())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dynamics given by the RRE exhibits a so-called *transcritical bifurcation* as the parameter $\lambda$ crosses the critical value $\lambda_c = 1$:
    - For $\lambda < \lambda_c$ the equilibrium $c=0$ is stable and hence the disease always dies out.
    - For $\lambda > \lambda_c$ the equlibrium $c=0$ is unstable, but the equilibirum $c_\infty = (\lambda - 1)/\lambda$ is stable. Thus the disease will prevail and the share of infectious nodes converges to $c_\infty$.

    In this notebook we have examined the SIS model on a random network using the CNVM package.
    We have found that in this system an *epidemic threshold* occurs, i.e., there is a critical infection rate that separates the regimes of the disease dying out and surviving.
    """)
    return


if __name__ == "__main__":
    app.run()
