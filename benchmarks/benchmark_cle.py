import time

from sponet import CNVMParameters, sample_cle


def main():
    r = [[0, 0.8, 0.2], [0.2, 0, 0.8], [0.8, 0.2, 0]]
    r_tilde = 0.01
    num_agents = 1000
    num_samples = 3200
    delta_t = 0.001
    t_max = 50
    t_eval = 51

    params = CNVMParameters(num_agents=num_agents, r=r, r_tilde=r_tilde)

    # compilation
    t, c = sample_cle(params, [0.2, 0.3, 0.5], 1, 16, 0.05, 11)

    start = time.time()
    t, c = sample_cle(params, [0.2, 0.3, 0.5], t_max, num_samples, delta_t, t_eval)
    end = time.time()

    print(f"Took {end - start} s")
    print(f"t.shape = {t.shape}")
    print(f"c.shape = {c.shape}")


if __name__ == "__main__":
    # took 5.0 s
    main()
