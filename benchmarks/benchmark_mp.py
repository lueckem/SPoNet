import time

import sponet.network_generator as ng
from sponet import CNVMParameters
from sponet.collective_variables import OpinionShares
from sponet.multiprocessing import sample_many_runs
from sponet.states import sample_states_uniform


def main():
    num_opinions = 2
    num_agents = 10000
    network = ng.BarabasiAlbertGenerator(num_agents, m=3)()
    params = CNVMParameters(
        num_opinions=num_opinions, network=network, r=1, r_tilde=0.01
    )
    t_max = 1000
    len_output = 1001
    x_init = sample_states_uniform(num_agents, num_opinions, 1)
    num_runs = 200
    n_jobs = 4
    cv = OpinionShares(num_opinions, normalize=True)

    start = time.time()
    sample_many_runs(
        params,
        x_init,
        t_max,
        len_output,
        num_runs,
        n_jobs=n_jobs,
        collective_variable=cv,
        progress_bar=True,
    )
    end = time.time()
    print(f"Took {end - start} s")


if __name__ == "__main__":
    main()
