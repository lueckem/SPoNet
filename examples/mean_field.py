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
    ### Mean-field limits

    This notebook discusses the behavior of the CNVM as we let the number of nodes $N$ increase.
    For certain networks, e.g., complete and Erdös-Rényi networks, it is known that the dynamics of the CNVM converges to a mean-field limit given by an ordinary differential equation (ODE) of the opinion shares.
    This ODE is also called mean-field equation or reaction-rate equation (RRE) and is defined as

    $$ \frac{d}{dt} c(t) = \sum_{\substack{m,n=1 \\ m\neq n}}^M c_m(t) (r_{mn} c_n(t) + \tilde{r}_{mn})(e_n - e_m), \quad \text{(RRE)} $$

    where $c(t)=(c_1(t), \dots, c_M(t))$ is the vector of opinion shares (e.g., $c_1(t) \in [0,1]$ is the share of nodes in state $1$), and $e_n$ is the $n$-th unit vector.
    (The proof can be found for example in [[Lücke et al., 2022]](https://arxiv.org/abs/2210.02934).)

    We show how to investigate this phenomenon using the CNVM package.
    First we do the necessary imports.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    from matplotlib import pyplot as plt

    from sponet import CNVMParameters, sample_many_runs, calc_rre_traj
    from sponet.collective_variables import OpinionShares
    from sponet.network_generator import ErdosRenyiGenerator

    return (
        CNVMParameters,
        ErdosRenyiGenerator,
        OpinionShares,
        calc_rre_traj,
        np,
        plt,
        sample_many_runs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will use the following example parameters.
    """
    )
    return


@app.cell
def _(OpinionShares, np):
    num_opinions = 3
    t_max = 100
    t_eval = 1001  # number of snapshots to store

    r = np.array([[0, 0.8, 0.2], [0.2, 0, 0.8], [0.8, 0.2, 0]])
    r_tilde = 0.01 * np.array([[0, 0.9, 0.7], [0.7, 0, 0.9], [0.9, 0.7, 0]])

    cv = OpinionShares(num_opinions, normalize=True)
    return cv, num_opinions, r, r_tilde, t_eval, t_max


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we conduct a statistical analysis of the CNVM for Erdös-Rényi (ER) networks ($p=0.1$) of different sizes.
    We sample multiple realizations for networks of sizes $N=200,1000,5000$ using the `sample_many_runs` function.
    The initial state is picked such that 40\% of nodes have state 0, 20\% state 1, and 40\% state 2.
    Moreover, we use a `NetworkGenerator` object to generate a new ER network for each sample.

    (If the following code block takes long on your machine, try reducing the `num_samples`.)
    """
    )
    return


@app.cell
def _(
    CNVMParameters,
    ErdosRenyiGenerator,
    cv,
    np,
    num_opinions,
    r,
    r_tilde,
    sample_many_runs,
    t_eval,
    t_max,
):
    num_agents = [200, 1000, 5000]
    num_samples = [500, 100, 10]
    c_list = []
    for n_a, n_samples in zip(num_agents, num_samples):
        network_gen = ErdosRenyiGenerator(n_a, p=0.1)
        params = CNVMParameters(
            num_opinions=num_opinions,
            network_generator=network_gen,
            r=r,
            r_tilde=r_tilde,
        )
        x_init = np.zeros(n_a)
        x_init[: int(0.2 * n_a)] = 1
        x_init[int(0.6 * n_a) :] = 2
        t, c = sample_many_runs(
            params, x_init, t_max, t_eval, n_samples, collective_variable=cv
        )
        c_list.append(c)
    return c_list, num_agents, params, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The CNVM package offers the function `calc_rre_traj` that calculates the solution of the mean-field ODE for us.
    """
    )
    return


@app.cell
def _(calc_rre_traj, np, params, t, t_max):
    c_init = np.array([0.4, 0.2, 0.4])
    t_rre, c_rre = calc_rre_traj(params, c_init, t_max, t_eval=t)
    return c_rre, t_rre


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Finally, we compare the results from our simulations to the solution of the RRE.
    As the theory suggests, a larger number of nodes yields a closer match between the average simulation and the RRE.
    """
    )
    return


@app.cell
def _(c_list, c_rre, np, num_agents, plt, t, t_rre):
    for _n_a, _c in zip(num_agents, c_list):
        plt.plot(t, np.mean(_c[:, :, 0], axis=0), label=f"{_n_a}")
    plt.plot(t_rre, c_rre[:, 0], "k", label="RRE")
    plt.grid()
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("$c_1$")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
