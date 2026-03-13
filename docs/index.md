# Spreading Processes on Networks (SPoNet) 

This package provides an efficient implementation of popular discrete-state spreading processes on networks of interacting *agents*.
They can be used to simulate how opinions about certain issues develop over time within a population, or how an infectious disease spreads.
The simulation loop is just-in-time compiled using `numba`, which makes performance comparable with compiled languages like C++.

## Installation
The package requires a Python version between 3.10 and 3.14.
Install from the PyPI repository:
```
pip install sponet
```
or get the latest version directly from GitHub:
```
pip install git+https://github.com/lueckem/SPoNet
```

## Available models

- [Continuous-time noisy voter model (CNVM)](models/cnvm.md)
- [Continuous-time noisy threshold model (CNTM)](models/cntm.md)

## Examples
Check out the examples in the [repository](https://github.com/lueckem/SPoNet/tree/main/examples).
They are [marimo](https://docs.marimo.io/) notebooks.
You can install marimo on your machine (`pip install marimo`), clone the repository, and launch via `marimo edit`.
Alternatively, you can also view the notebooks in your browser using the [molab](https://docs.marimo.io/guides/molab/) web interface by following these links:

- [Notebook CNVM tutorial](https://molab.marimo.io/github/lueckem/SPoNet/blob/main/examples/tutorial_cnvm.py)
- [Notebook SIS model](https://molab.marimo.io/github/lueckem/SPoNet/blob/main/examples/SIS_model.py)

## Reference

- [CNVM](reference/cnvm.md)
- [CNTM](reference/cntm.md)
- [Network generators](reference/network_generators.md): generate (random) networks
- [Collective variables](reference/collective_variables.md): aggregate state information
- [Sampling states](reference/states.md): sample system states (e.g., as initial states for simulations)
- [Multiprocessing](reference/multiprocessing.md): run many simulations in parallel
