import marimo

__generated_with = "0.19.2"
app = marimo.App()


@app.cell
def _():
    # for running the notebook
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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

    Let us start by doing the necessary imports and defining the model.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import networkx as nx
    from matplotlib import pyplot as plt

    from sponet import CNVMParameters, CNVM
    from sponet.collective_variables import OpinionShares
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
def _(CNVM, CNVMParameters, OpinionShares, np, nx):
    num_opinions = 2  # opinion 1 represents 'S', opinion 2 represents 'I'
    num_agents = 1000
    _infection_rate = 0.5
    _r = np.array([[0, _infection_rate], [0, 0]])
    r_tilde = np.array([[0, 0], [1, 0]])
    network = nx.erdos_renyi_graph(num_agents, p=0.1)
    params = CNVMParameters(
        num_opinions=num_opinions, network=network, r=_r, r_tilde=r_tilde
    )
    model = CNVM(params)
    cv = OpinionShares(
        num_opinions, normalize=True
    )  # for measuring the percentage of infectious nodes
    return cv, model, num_agents, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The behavior of the SIS model heavily depends on the underlying network structure.
    We have chosen a random Erdös-Renyi graph in this example because it is well understood.

    Now we define the simulation parameters, let the model run, and plot the results.
    We start with 30% infectious nodes and plot the evolution of the share of infectious nodes.
    """
    )
    return


@app.cell
def _(np, num_agents):
    t_max = 100
    t_eval = 1001
    x_init = np.zeros(num_agents)  # initial state
    x_init[: int(num_agents * 0.3)] = 1
    np.random.shuffle(x_init)
    return t_eval, t_max, x_init


@app.cell
def _(cv, model, plt, t_eval, t_max, x_init):
    _t, _x = model.simulate(t_max=t_max, x_init=x_init, t_eval=t_eval)
    _c = cv(_x)  # calculate the share of infectious nodes
    plt.plot(_t, _c[:, 1])
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("percentage infected")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The above plot shows that the disease dies out quickly. After a short time all the nodes have state (S).
    This is because the infection rate was rather small ($\lambda=0.5$).
    Let us investigate the dynamics for a larger infection rate of $\lambda=2$.
    """
    )
    return


@app.cell
def _(cv, model, np, plt, t_eval, t_max, x_init):
    _infection_rate = 2
    _r = np.array([[0, _infection_rate], [0, 0]])
    model.update_rates(r=_r)
    _t, _x = model.simulate(t_max=t_max, x_init=x_init, t_eval=t_eval)
    _c = cv(_x)
    plt.plot(_t, _c[:, 1])
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("percentage infected")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now the disease did not die out. After a short transient phase, the percentage of infectious nodes stabilizes at around 50%.

    Let us conduct a statistical analysis of this behavior for different infection rates.
    In the following code block we perform many simulations of the SIS model.
    (If this takes too long on your machine, try reducing the number of samples by modifying the `num_runs` parameter.)
    """
    )
    return


@app.cell
def _(cv, np, params, sample_many_runs, t_eval, t_max, x_init):
    infection_rates = [0.8, 0.9, 1.0, 1.1, 1.2]
    c_results = []
    for _i_r in infection_rates:
        _r = np.array([[0, _i_r], [0, 0]])
        params.change_rates(r=_r)
        t, _c = sample_many_runs(
            params=params,
            initial_states=x_init,
            t_max=t_max,
            t_eval=t_eval,
            num_runs=100,
            collective_variable=cv,
            n_jobs=-1,
        )
        c_results.append(_c)
    return c_results, infection_rates, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We plot the average share of infectious nodes and the probability that the disease survived up to time $t=100$ (i.e., there is at least one infectious node present).
    """
    )
    return


@app.cell
def _(c_results, infection_rates, np, plt, t):
    for _i_r, c_r in zip(infection_rates, c_results):
        plt.plot(t, np.mean(c_r[:, :, 1], axis=0), label=f"rate={_i_r}")
    plt.legend()
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("percentage infected")
    plt.show()
    return


@app.cell
def _(c_results, infection_rates, np, plt):
    persistence_probabilities = [
        np.linalg.norm(c_r[:, -1, 1], 0) / c_r.shape[1] for c_r in c_results
    ]

    plt.plot(infection_rates, persistence_probabilities, "--x")
    plt.ylabel("infection survival probability")
    plt.xlabel("infection rate")
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Apparently, the disease will likely die out if the infection rate is below 1, and will likely persist if the infection rate is above 1.
    This critical value is called the *epidemic threshold*.

    For this example (the SIS model on a sufficiently dense Erdös-Renyi random graph) it is known that the evolution of the share of infectious nodes $c(t)$ is given by the following mean-field ODE in the large population limit:

    $$ \frac{d}{dt} c(t) = -c(t) + \lambda (1 - c(t)) c(t). \qquad \text{(reaction-rate equation (RRE))} $$

    (See the notebook `mean_field.ipynb` or the paper [[Lücke et al., 2022]](https://arxiv.org/abs/2210.02934) for further information about the RRE.)

    The plot below shows that this ODE is already reasonably accurate for our finite size network.
    """
    )
    return


@app.cell
def _(c_results, calc_rre_traj, np, params, plt, t, t_max):
    t_rre, c_rre = calc_rre_traj(params, c_results[-1][0, 0], t_max)

    plt.plot(t_rre, c_rre[:, 1], "-k", label="RRE")
    plt.plot(t, np.mean(c_results[-1][:, :, 1], axis=0), label="model")
    plt.legend()
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("percentage infected")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The dynamics given by the RRE exhibits a so-called *transcritical bifurcation* as the parameter $\lambda$ crosses the critical value $\lambda_c = 1$:
    - For $\lambda < \lambda_c$ the equilibrium $c=0$ is stable and hence the disease always dies out.
    - For $\lambda > \lambda_c$ the equlibrium $c=0$ is unstable, but the equilibirum $c_\infty = (\lambda - 1)/\lambda$ is stable. Thus the disease will prevail and the share of infectious nodes converges to $c_\infty$.

    In this notebook we have examined the SIS model on a random network using the CNVM package.
    We have found that in this system an *epidemic threshold* occurs, i.e., there is a critical infection rate that separates the regimes of the disease dying out and surviving.
    """
    )
    return


if __name__ == "__main__":
    app.run()
