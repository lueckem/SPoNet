# Network Generators

A _Network Generator_ can be used to sample random networks for usage in one of the models.
Simply define the generator and call it to sample a network, for example:

```python
from sponet import ErdosRenyiGenerator

er_generator = ErdosRenyiGenerator(1000, 0.1)
network = er_generator()
```

The available generators are listed below.

::: sponet.network_generator
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ErdosRenyiGenerator
        - RandomRegularGenerator
        - BarabasiAlbertGenerator
        - WattsStrogatzGenerator
        - StochasticBlockGenerator
        - GridGenerator
        - BinomialWattsStrogatzGenerator
        - BianconiBarabasiGenerator
