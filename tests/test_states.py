import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

import sponet.collective_variables as cvs
import sponet.states as ss


def assert_states_valid(
    states: NDArray, num_agents: int, num_opinions: int, num_states: int | None
):
    if num_states is None or num_states == 1:
        assert states.shape == (num_agents,)
    else:
        assert states.shape == (num_states, num_agents)
    assert np.issubdtype(states.dtype, np.integer)
    assert np.all(states >= 0)
    assert np.all(states < num_opinions)


@pytest.mark.parametrize(
    "num_agents,num_opinions,num_states",
    [
        (100, 3, None),
        (100, 4, 1),
        (100, 3, 30),
        (50, 2, 20),
    ],
)
def test_sample_states_uniform(num_agents, num_opinions, num_states):
    if num_states is None:
        states = ss.sample_states_uniform(num_agents, num_opinions)
    else:
        states = ss.sample_states_uniform(num_agents, num_opinions, num_states)
    assert_states_valid(states, num_agents, num_opinions, num_states)


@pytest.mark.parametrize(
    "num_agents,num_opinions,num_states",
    [
        (100, 3, None),
        (100, 4, 1),
        (100, 3, 30),
        (50, 2, 20),
    ],
)
def test_sample_states_uniform_shares(num_agents, num_opinions, num_states):
    if num_states is None:
        states = ss.sample_states_uniform_shares(num_agents, num_opinions)
    else:
        states = ss.sample_states_uniform_shares(num_agents, num_opinions, num_states)
    assert_states_valid(states, num_agents, num_opinions, num_states)


@pytest.mark.parametrize(
    "num_agents,target_shares,num_states",
    [
        (100, [0.2, 0.3, 0.5], None),
        (100, [0.1, 0, 0.8, 0.1], 1),
        (100, [0.2, 0.3, 0.5], 30),
        (50, [0.7, 0.3, 0], 20),
    ],
)
def test_sample_states_target_shares(num_agents, target_shares, num_states):
    if num_states is None:
        states = ss.sample_states_target_shares(num_agents, target_shares)
    else:
        states = ss.sample_states_target_shares(num_agents, target_shares, num_states)
    num_opinions = len(target_shares)
    assert_states_valid(states, num_agents, num_opinions, num_states)

    cv = cvs.OpinionShares(num_opinions, normalize=True)
    assert np.allclose(cv(states), target_shares)


@pytest.mark.parametrize(
    "num_agents,num_opinions,num_states",
    [
        (100, 3, None),
        (100, 4, 1),
        (100, 3, 30),
        (50, 2, 20),
    ],
)
def test_sample_states_local_clusters(num_agents, num_opinions, num_states):
    network = nx.barabasi_albert_graph(num_agents, 3)
    if num_states is None:
        states = ss.sample_states_local_clusters(network, num_opinions)
    else:
        states = ss.sample_states_local_clusters(
            network, num_opinions, num_states, 4, 2
        )
    assert_states_valid(states, num_agents, num_opinions, num_states)


@pytest.mark.parametrize(
    "num_agents,target_shares,num_states",
    [
        (100, [0.2, 0.3, 0.5], None),
        (100, [0.1, 0, 0.8, 0.1], 10),
        (100, [0.2, 0.3, 0.5], 30),
        (1000, [1.0, 0], 1),
    ],
)
def test_sample_states_target_cvs(num_agents, target_shares, num_states):
    num_opinions = len(target_shares)
    cv = cvs.OpinionShares(num_opinions, normalize=True)
    target_shares = np.array(target_shares)
    if num_states is None:
        states = ss.sample_states_target_cvs(
            num_agents, num_opinions, cv, target_shares
        )
    else:
        states = ss.sample_states_target_cvs(
            num_agents, num_opinions, cv, target_shares, num_states
        )
    assert_states_valid(states, num_agents, num_opinions, num_states)
    assert np.allclose(cv(states), target_shares)


def test_build_state_by_degree_valid():
    num_agents = 100
    opinion_shares = np.array([0.2, 0.3, 0.5])
    opinion_order = np.array([1, 0, 2])
    network = nx.barabasi_albert_graph(num_agents, 3)
    x = ss.build_state_by_degree(network, opinion_shares, opinion_order)
    assert x.shape == (num_agents,)
    assert np.issubdtype(x.dtype, np.integer)
    assert np.all(x >= 0)
    assert np.all(x < 3)


def test_build_state_by_degree_example():
    network = nx.Graph()
    network.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
        ]
    )  # degrees = [4, 3, 2, 2, 1]

    opinion_shares = np.array([0.4, 0.4, 0.2])
    opinion_order = np.array([1, 0, 2])
    x = ss.build_state_by_degree(network, opinion_shares, opinion_order)
    assert np.all(x == np.array([1, 1, 0, 0, 2]))
