from dataclasses import dataclass
import networkx as nx


@dataclass()
class CNTMParameters:
    """
    Container for the parameters of the Threshold Model.

    At the rate r, each node evaluates to change their opinion from its current
    opinion m=0,1 to the other opinion n=1-m. It changes the opinion if the
    percentage of neighbors of opinion n exceeds the threshold_mn.

    Additionally, each node changes its state randomly at rate r_tilde (noise).
    """

    network: nx.Graph
    r: float
    r_tilde: float
    threshold_01: float
    threshold_10: float

    def __post_init__(self):
        self.num_agents = self.network.number_of_nodes()
        self.num_opinions = 2
