# Spreading Processes on Networks (SPoNet)

[![build](https://github.com/lueckem/SPoNet/actions/workflows/build.yml/badge.svg)](https://github.com/lueckem/SPoNet/actions/workflows/build.yml)

This package provides an efficient implementation of popular discrete-state spreading processes on networks of interacting *agents*.
They can be used to simulate how opinions about certain issues develop over time within a population, or how an infectious disease spreads.
The simulation loop is just-in-time compiled using `numba`, which makes performance comparable with compiled languages like C++.

Available models:
- continuous-time noisy voter model (CNVM)
- continuous-time noisy threshold model (CNTM)


## Installation
The package requires Python 3.9, 3.10, 3.11, or 3.12.
Install from the PyPI repository:
```
pip install sponet
```
or get the latest version directly from GitHub:
```
pip install git+https://github.com/lueckem/SPoNet
```

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

### Basic Usage
First define the model parameters:

```python
from sponet import CNVMParameters
import numpy as np
import networkx as nx

num_nodes = 100
r = np.array([[0, .8], [.2, 0]])
r_tilde = np.array([[0, .1], [.2, 0]])
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
from sponet import CNVM

x_init = np.random.randint(0, 2, num_nodes)
model = CNVM(params)
t, x = model.simulate(t_max=50, x_init=x_init)
```
The output `t` contains the time points of state jumps and `x` the system states after each jump.

A more detailed overview of the package can be found in the jupyter notebook [*examples/tutorial_cnvm.ipynb*](examples/tutorial_cnvm.ipynb).
Moreover, the behavior of the CNVM in the mean-field limit is discussed in [*examples/mean_field.ipynb*](examples/mean_field.ipynb).
In the notebook [*examples/SIS-model.ipynb*](examples/SIS-model.ipynb) the existence of an epidemic threshold for the SIS model in epidemiology is demonstrated.

### Implementation details

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

### Basic Usage
First define the model parameters:

```python
from sponet import CNTMParameters
import numpy as np
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

x_init = np.random.randint(0, 2, num_nodes)
model = CNTM(params)
t, x = model.simulate(t_max=50, x_init=x_init)
```
The output `t` contains the time points of state jumps and `x` the system states after each jump.
