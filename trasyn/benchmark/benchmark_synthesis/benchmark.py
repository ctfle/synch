import numpy as np
from trasyn.benchmark.utils import generate_unitaries, benchmark_on_random_unitaries

SEED = 42
np.random.seed(SEED)


def run_benchmark(args):
    return benchmark_on_random_unitaries(**args)


if __name__ == "__main__":
    n_unitaries = 100
    unitaries = generate_unitaries(n_unitaries)
    min_budget = 1
    max_budget = 8
    num_attempts = 5
    num_samples = 20000
    max_partition_value = 10
    save_dir = f"./test_benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_{num_attempts}_num_samples_{num_samples}_max_partition_value={max_partition_value}"
    gate_set_cost_25 = "tqshxyz"

    benchmark_on_random_unitaries(
        unitaries=unitaries,
        budgets=np.arange(min_budget, max_budget, 0.5),
        load_dir="../../assets/" + gate_set_cost_25,
        save_dir=save_dir + "/" + gate_set_cost_25,
        seed=SEED,
        costs={"t": 1.0, "q": 2.5},
        num_attempts=num_attempts,
        num_samples=num_samples,
        max_partition_value=max_partition_value,
    )
