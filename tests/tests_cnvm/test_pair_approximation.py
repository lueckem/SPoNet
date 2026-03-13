import networkx as nx
import numpy as np

from sponet.cnvm.approximations.pair_approximation import calc_pair_approximation_traj
from sponet.cnvm.parameters import CNVMParameters
from sponet.collective_variables import Interfaces


def test_calc_pair_approximation_traj():
    num_opinions = 2
    num_agents = 1000
    r = np.array([[0, 1], [1.1, 0]])
    r_tilde = 0.01
    network = nx.barabasi_albert_graph(num_agents, 3)

    params = CNVMParameters(
        num_opinions=num_opinions,
        network=network,
        r=r,
        r_tilde=r_tilde,
    )

    t_max = 100
    num_time_steps = 10000
    x_0 = np.zeros(num_agents)
    x_0[:100] = 1
    c_0 = 0.1
    s_0 = 0.5 * Interfaces(network, True)(np.array([x_0]))[0, 0]
    mean_degree = np.mean([d for _, d in network.degree()])  # type: ignore

    t, c_pa = calc_pair_approximation_traj(
        params,
        c_0,
        s_0,
        mean_degree,  # type: ignore
        t_max,
        t_eval=np.linspace(0, t_max, num_time_steps + 1),
    )

    assert c_pa.shape == (10001, 2)
    assert c_pa[0, 0] == c_0
    assert c_pa[0, 1] == s_0
    assert np.allclose(t, np.linspace(0, t_max, num_time_steps + 1))
