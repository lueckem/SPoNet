# Continuous-time noisy threshold model (CNTM)

This page explains what the CNTM is and how it is implemented in this package.
For the code reference see [here](../reference/cntm.md).

## About the CNTM
On a network (undirected simple graph) of $N$ nodes, each node $i$ has one of two opinions $x_i \in \{0, 1\}$.
At the rate $r \geq 0$, each node evaluates to change their opinion from its current
opinion $m\in \{0, 1\}$ to the other opinion $n=1-m$. It changes the opinion if the
percentage of neighbors of opinion n exceeds the threshold $b_{m,n}$.
Additionally, each node changes its state randomly at rate $\tilde{r} \geq 0$ (noise).
Hence, the rate at which node $i$ switches from opinion $m$ to opinion $n$ is

$$ r \ \delta_{\left( \frac{d_{i,n}(x)}{d_{i}} \geq b_{m,n} \right)} + \tilde{r} $$

where $d_{i,n}(x)$ denotes the number of neighbors of node $i$ with opinion $n$ and $d_i$ is the degree of node $i$.
Thus, in contrast to the CNVM, the CNTM assumes that a switch to a different opinion only occurs
if that opinion is already sufficiently established in the neighborhood.

## Basic Usage
First define the model parameters:

```python
from sponet import CNTMParameters
import networkx as nx

num_nodes = 100
r = 1
r_tilde = 0.1
threshold_01 = 0.5
threshold_10 = 0.3
network = nx.erdos_renyi_graph(n=num_nodes, p=0.1)

params = CNTMParameters(
    network=network,
    r=r,
    r_tilde=r_tilde,
    threshold_01=threshold_01,
    threshold_10=threshold_10,
)
```
Then simulate the model, starting in state `x_init`:

```python
from sponet import CNTM
import numpy as np

x_init = np.random.randint(0, 2, num_nodes)
model = CNTM(params)
t, x = model.simulate(t_max=50, x_init=x_init)
```
The output `t` contains the time points of state jumps and `x` the system states after each jump.
