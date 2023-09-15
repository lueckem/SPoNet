from __future__ import annotations
from typing import Protocol
import networkx as nx
import pickle


class Parameters(Protocol):
    num_opinions: int
    num_agents: int
    network: nx.Graph | None


def save_params(filename: str, params: Parameters):
    """
    Save parameters as pickled file.

    Parameters
    ----------
    filename : str
    params : CNVMParameters
    """
    with open(filename, "wb") as file:
        pickle.dump(params, file)


def load_params(filename: str) -> Parameters:
    """
    Load parameters from pickled file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    CNVMParameters
    """
    with open(filename, "rb") as file:
        return pickle.load(file)
