# Collective Variables

A _Collective Variable_ can be used to aggregate information of the microscropic system state into a macroscopic description.
The simplest example are the `OpinionShares` that count how often each discrete state occurs:

```python
import numpy as np
from sponet import OpinionShares

num_agents = 100
x = np.random.randint(0, 2, num_agents)

cv = OpinionShares(num_opinions=2)
c = cv(x)
```

In the example above, `x` contains the state of each agent, e.g., `x = [0, 0, 1, 0, 1, ...]`, and `c` contains the counts of zeros and ones, e.g., `c = [52, 48]`.

The available collective variables are listed below.

::: sponet.collective_variables
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - OpinionShares
        - DegreeWeightedOpinionShares
        - OpinionSharesByDegree
        - Interfaces
        - Propensities
        - CompositeCollectiveVariable
