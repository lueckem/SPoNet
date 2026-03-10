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

- continuous-time noisy voter model (CNVM)
- continuous-time noisy threshold model (CNTM)

