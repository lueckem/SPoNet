# Continuous-time noisy voter model (CNVM)

This page explains what the CNVM is and how it is implemented in this package.
For the code reference see [here](../reference/cnvm.md).

## About the CNVM
Let a network (undirected simple graph) of $N$ nodes be given. The nodes represent agents and the edges social interactions. 
Each node is endowed with one of $M$ discrete opinions. Thus, the system state is given by a vector $x \in \{1,\dots,M\}^N$, where $x_i$ describes the opinion of node $i$.
Each node's opinion $x_i \in \{1,\dots,M\}$ changes over time according to a continuous-time Markov chain (Markov jump process).
Given the current system state $x$, the generator matrix $Q^i$ of the continuous-time Markov chain associated with node $i$ is defined as

$$ Q^i \in \mathbb{R}^{M \times M},\quad (Q^i)_ {m,n} := r_{m,n} \frac{d_ {i,n}(x)}{(d_i)^\alpha} + \tilde{r}_ {m,n},\ m\neq n, $$

where $d_{i,n}(x)$ denotes the number of neighbors of node $i$ with opinion $n$ and $d_i$ is the degree of node $i$. The matrices $r, \tilde{r} \in \mathbb{R}^{M \times M}$ and $\alpha \in \mathbb{R}$ are model parameters.

Thus, the transition rates $(Q^i)_ {m,n}$ consist of two components.
The first component $r_{m,n} d_{i,n}(x)/ (d_i)^\alpha$ describes at which rate node $i$ gets "infected" with opinion $n$ by nodes in its neighborhood.
The second part $\tilde{r}_{m,n}$ describes transitions that are independent from the neighborhood (e.g., noise).

The parameter $\alpha$ can be used to tune the type of interaction. For $\alpha=1$ the transition rates are normalized because $d_{i,n}(x)/d_i \in [0,1]$.
The setting $\alpha=0$ however yields a linear increase of the transition rates with the number of "infected" neighbors, and is often used in epidemic modeling, e.g., the contact process or SIS model.

The network itself is static, i.e., the nodes and edges do not change over time.

## Basic Usage
First define the model parameters.
Here we define a simple example with 2 opinions and a random network.

```python
from sponet import CNVMParameters
import networkx as nx

num_nodes = 100
r = [[0, .8], [.2, 0]]
r_tilde = [[0, .1], [.2, 0]]
network = nx.erdos_renyi_graph(n=num_nodes, p=0.1)

params = CNVMParameters(
    num_opinions=2,
    network=network,
    r=r,
    r_tilde=r_tilde,
    alpha=1,
)
```

Then simulate the model, starting in state `x_init`:

```python
import numpy as np
from sponet import CNVM

x_init = np.random.randint(0, 2, num_nodes)
model = CNVM(params)
t, x = model.simulate(t_max=50, x_init=x_init)
```
The output `t` contains the time points of state jumps and `x` the system states after each jump.

## Implementation details

After a node switches its opinion, the system state $x$ changes and hence all the generator matrices $Q^i$ may change as well.
We apply a Gillespie-like algorithm to generate statistically correct samples of the process.
We start a Poisson clock for each possible transition and as soon as the first transition occurs we modify the generator matrices and reset all the clocks.
To do this efficiently, it is advantageous to transform the rate matrices $r$ and $\tilde{r}$ into an equivalent format consisting of base rates $r_0, \tilde{r}_0 > 0$ and probability matrices $p, \tilde{p} \in [0, 1]^{M\times M}$ such that

$$ r_{m,n} = r_0 p_ {m,n}, \quad \tilde{r}_ {m,n} = \tilde{r}_ 0 \tilde{p}_ {m,n} / M. $$

Furthermore, we define the cumulative rates

$$ \lambda := \sum_{i=1}^N r_0 d_i^{(1-\alpha)},\quad \tilde{\lambda} := N \tilde{r}_0,\quad \hat{\lambda} := \lambda + \tilde{\lambda}. $$

Then the simulation loop is given by

1. Draw time of next jump event from exponential distribution $\exp(\hat{\lambda})$. Go to 2.
2. With probability $\lambda / \hat{\lambda}$ the event is due to infection, in which case go to 3.
Else it is due to noise, go to 4.
3. Draw agent $i$ from $\{1,\dots,N\}$ according to distribution $\mathbb{P}(i = j) = r_0 d_j^{(1-\alpha)} / \lambda$. Let $m$ denote the state of agent $i$.
Draw $n$ from $\{1,\dots,M\}$ according to $\mathbb{P}(n = k) = d_{i,k}(x) / d_i$.
With probability $p_{m,n}$ agent $i$ switches to state $n$. Go back to 1.
4. Draw $i$ from $\{1,\dots,N\}$ and $n$ from $\{1,\dots,M\}$ uniformly. Let $m$ denote the state of agent $i$.
With probability $\tilde{p}_{m,n}$ agent $i$ switches to state $n$. Go back to 1.
