import networkx as nx
import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


@njit(cache=True)
def argmatch(x_ref: NDArray, x: NDArray) -> NDArray:
    """
    Find indices such that |x[indices] - x_ref| = min!

    Parameters
    ----------
    x_ref : NDArray
        1D, sorted
    x : NDArray
        1D, sorted
    Returns
    -------
    NDArray
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


def mask_subsequent_duplicates(x: NDArray) -> NDArray:
    """
    Calculate mask that removes subsequent duplicates.

    For example if x=[1,1,2,2,3,1,3,3] then x[mask]=[1,2,3,1,3].
    If x has more than one dimensions, the duplicates are removed w.r.t. the first axis.
    For example if x=[[1,1],[1,1],[2,2]] then x[mask]=[[1,1],[2,2]].

    Parameters
    ----------
    x : NDArray
        1D or 2D array.

    Returns
    -------
    NDArray
    """
    if x.ndim == 1:
        mask = x[:-1] != x[1:]
    elif x.ndim == 2:
        mask = np.any(x[:-1] != x[1:], axis=1)
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")
    mask = np.concatenate((np.array([True]), mask))

    return mask


def calculate_neighbor_list(network: nx.Graph) -> list[NDArray]:
    """
    Calculate list of neighbors.

    The i-th element of the list is a numpy array containing the
    node-indices of the neighbors of node i.

    Parameters
    ----------
    network : nx.Graph

    Returns
    -------
    list[NDArray]
    """
    neighbor_list = []
    for i in network.nodes():
        neighbor_list.append(np.array(list(network.neighbors(i)), dtype=int))
    return neighbor_list


@njit(cache=True)
def store_snapshot_linspace(
    t: float,
    t_store: float,
    previous_t: float,
    x: NDArray,
    previous_agent: int,
    previous_opinion: int,
    x_traj: list[NDArray],
    t_traj: list[float],
) -> None:
    """
    Store either the current snapshot in `t_traj` and `x_traj` or the previous one,
    depending on which is closer to `t_store`.

    Parameters
    ----------
    t : float
        Current time.
    t_store : float
        Desired time.
    previous_t : float
        Time of previous snapshot.
    x : NDArray
        System state.
    previous_agent : int
        Agent that switched from previous to current snapshot.
    previous_opinion : int
        Previous opinion of the agent before the switch.
    x_traj : list[NDArray]
    t_traj : list[float]
    """
    x_store = x.copy()
    if t - t_store <= abs(t_store - previous_t):  # t is closer
        x_traj.append(x_store)
        t_traj.append(t)
    else:  # previous_t is closer
        x_store[previous_agent] = previous_opinion  # revert to previous state
        x_traj.append(x_store)
        t_traj.append(previous_t)


def counts_from_shares(shares: ArrayLike, num_agents: int) -> NDArray:
    """
    Convert shares like [0.2, 0.3, 0.5] into counts like [20, 30, 50].

    Parameters
    ----------
    shares : NDArray
        Shares with shape (num_opinions,) or (num_states, num_opinions).
    num_agents : int

    Returns
    -------
    NDArray
        Counts with shape (num_opinions,) or (num_states, num_opinions).
    """
    shares = np.array(shares) * num_agents
    if shares.ndim == 1:
        return _counts_from_shares_1d(shares, num_agents)
    else:
        counts = np.zeros_like(shares, dtype=int)
        for i in range(counts.shape[0]):
            counts[i] = _counts_from_shares_1d(shares[i], num_agents)
        return counts


def _counts_from_shares_1d(shares: NDArray, num_agents: int) -> NDArray:
    counts = np.floor(shares)
    deltas = counts - shares  # <= 0
    counts = counts.astype(int)
    num_incr = num_agents - np.sum(counts)
    idx_incr = np.argpartition(deltas, num_incr)[:num_incr]
    counts[idx_incr] += 1
    return counts
