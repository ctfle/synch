import numpy as np
from trasyn.benchmark.utils import generate_unitaries, benchmark_on_random_unitaries

SEED = 42
np.random.seed(SEED)

def run_benchmark(args):
    return benchmark_on_random_unitaries(**args)


if __name__ == "__main__":
    n_unitaries = 100
    unitaries = generate_unitaries(n_unitaries)
    min_budget = 10
    max_budget = 15
    num_attempts = 5
    num_samples = 20000
    max_partition_value = 10
    save_dir = f"./benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_{num_attempts}_num_samples_{num_samples}_max_partition_value={max_partition_value}"
    gate_set_cost_25 = "merged/tqshxyz_tequiv_large_cost_2.5_all"
    gate_set_t = "tshxyz_tequiv_large"

    tasks = [
        dict(
            unitaries=unitaries,
            budgets=np.arange(min_budget, max_budget),
            load_dir="../../assets/" + gate_set_t,
            save_dir=save_dir + "/" + gate_set_t,
            seed=SEED,
            costs={"T": 1.0},
            num_attempts=num_attempts,
            num_samples=num_samples,
            max_partition_value=max_partition_value,
        ),
        dict(
            unitaries=unitaries,
            budgets=np.arange(min_budget, max_budget, 0.5),
            load_dir="../../assets/" + gate_set_cost_25,
            save_dir=save_dir + "/" + gate_set_cost_25,
            seed=SEED,
            costs={"T": 1.0, "sqrtT": 2.5},
            num_attempts=num_attempts,
            num_samples=num_samples,
            max_partition_value=max_partition_value
        ),
    ]

    for task in tasks:
        run_benchmark(task)
