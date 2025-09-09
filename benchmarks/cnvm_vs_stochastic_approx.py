import time

from sponet import CNVMParameters
from sponet.cnvm.approximations.stochastic_approximation import (
    sample_stochastic_approximation,
)
from sponet.collective_variables import OpinionShares
from sponet.multiprocessing import sample_many_runs
from sponet.states import sample_states_uniform


def main():
    """CNVM model on a complete network vs stochastic approximation (Gillespie on the shares)."""
    num_opinions = 3
    num_agents = 100000
    params = CNVMParameters(
        num_opinions=num_opinions, num_agents=num_agents, r=1, r_tilde=0.01
    )
    t_max = 1000
    len_output = 1001
    x_init = sample_states_uniform(num_agents, num_opinions, 1)
    num_runs = 32
    n_jobs = 16
    cv = OpinionShares(num_opinions, normalize=True)
    c_init = cv(x_init)

    start = time.time()
    t, c = sample_many_runs(
        params,
        x_init,
        t_max,
        len_output,
        num_runs,
        n_jobs=n_jobs,
        collective_variable=cv,
    )
    end = time.time()
    print(c.shape)
    print(f"CNVM took {end - start} s")

    start = time.time()
    t, c = sample_stochastic_approximation(
        params, c_init, t_max, len_output - 1, num_runs
    )
    end = time.time()
    print(c.shape)
    print(f"Stoch. Approx. took {end - start} s")


if __name__ == "__main__":
    main()

# (32, 1001, 3)
# CNVM took 7.788949489593506 s
# (32, 1001, 3)
# Stoch. Approx. took 19.041971445083618 s
