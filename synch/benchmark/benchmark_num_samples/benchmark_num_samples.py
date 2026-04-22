import numpy as np
from synch.benchmark.utils import generate_unitaries, benchmark_on_random_unitaries

SEED = 42
np.random.seed(SEED)
# benchmark the influence of the num_samples chosen


def run_benchmark(args):
    return benchmark_on_random_unitaries(**args)


if __name__ == "__main__":
    n_unitaries = 100
    unitaries = generate_unitaries(n_unitaries)
    num_attempts = 5
    samples = [100, 1000, 10_000, 20_000, 30_000, 40_000, 50_000]
    max_partition_value = 10
    for budget in [16, 19]:
        if budget == 10:
            samples = [100, 1000, 10_000]
        elif budget == 13:
            samples = [100, 1000, 10_000, 20_000]
        elif budget == 16:
            samples = [20_000, 40_000, 50_000]
        else:
            samples = [1000, 20_000, 50_000, 80_000]

        for num_samples in samples:
            save_dir = f"./benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_{num_attempts}_num_samples_{num_samples}_max_partition_value_{max_partition_value}"
            gate_set_cost_25 = "merged/tqshxyz_tequiv_large_cost_2.5_all"
            gate_set_t = "tshxyz_tequiv_large"

            tasks = [
                dict(
                    unitaries=unitaries,
                    budgets=np.arange(budget, budget + 1),
                    load_dir="../../assets/" + gate_set_cost_25,
                    save_dir=save_dir + "/" + gate_set_cost_25,
                    seed=SEED,
                    costs={"T": 1.0, "sqrtT": 2.5},
                    num_attempts=num_attempts,
                    num_samples=num_samples,
                    max_partition_value=max_partition_value,
                ),
            ]

            for task in tasks:
                run_benchmark(task)
