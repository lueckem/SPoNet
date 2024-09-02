import numpy as np
from sponet import CNVMParameters, CNVM
import sponet.network_generator as ng
import time


def main():
    num_opinions = 2
    num_agents = 10000
    network = ng.BarabasiAlbertGenerator(num_agents, m=3)()
    params = CNVMParameters(
        num_opinions=num_opinions, network=network, r=1, r_tilde=0.01
    )
    t_max = 1000
    len_output = 1001
    bench_mark_time = 10  # in seconds

    model = CNVM(params)
    model.simulate(t_max=1, len_output=2)  # compile

    run_times = []
    b_start = time.time()
    while time.time() < b_start + bench_mark_time:
        start = time.time()
        t, x = model.simulate(t_max=t_max, len_output=len_output)
        end = time.time()
        run_times.append(end - start)

    print(f"Mean: {np.mean(run_times):.4f}s.")
    print(f"STD: {np.std(run_times):.4f}s.")
    print(f"Iterations: {len(run_times)}.")


if __name__ == "__main__":
    main()
