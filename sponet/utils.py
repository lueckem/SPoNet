import networkx as nx
import numpy as np
from numba import njit


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


def mask_subsequent_duplicates(x: np.ndarray) -> np.ndarray:
    """
    Calculate mask that removes subsequent duplicates.

    For example if x=[1,1,2,2,3,1,3,3] then x[mask]=[1,2,3,1,3].
    If x has more than one dimensions, the duplicates are removed w.r.t. the first axis.
    For example if x=[[1,1],[1,1],[2,2]] then x[mask]=[[1,1],[2,2]].

    Parameters
    ----------
    x : np.ndarray
        1D or 2D array.

    Returns
    -------
    np.ndarray
    """
    if x.ndim == 1:
        mask = x[:-1] != x[1:]
    elif x.ndim == 2:
        mask = np.any(x[:-1] != x[1:], axis=1)
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")
    mask = np.concatenate((np.array([True]), mask))

    return mask


def calculate_neighbor_list(network: nx.Graph):
    """
    Calculate list of neighbors.

    The i-th element of the list is a numpy array containing the
    node-indices of the neighbors of node i.

    Parameters
    ----------
    network : nx.Graph

    Returns
    -------
    List[np.ndarray]
    """
    neighbor_list = []
    for i in network.nodes():
        neighbor_list.append(np.array(list(network.neighbors(i)), dtype=int))
    return neighbor_list
