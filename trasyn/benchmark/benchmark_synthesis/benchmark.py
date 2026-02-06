import numpy as np
from trasyn.benchmark.utils import generate_unitaries, benchmark_on_random_unitaries, SEED

def run_benchmark(args):
    return benchmark_on_random_unitaries(**args)


if __name__ == "__main__":
    n_unitaries = 100
    unitaries = generate_unitaries(n_unitaries)
    min_budget = 10
    max_budget = 11
    num_attempts = 5
    num_samples = 10
    save_dir = f"./benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_{num_attempts}_num_samples_{num_samples}"
    gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"  # "tqshxyz_tequiv_short_cost_3"
    gate_set_cost_2 = "tqshxyz_tequiv_medium_cost_2"  # "tqshxyz_tequiv_short"
    gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
    gate_set_t = "tshxyz_tequiv_medium"

    tasks = [
        dict(
            unitaries=unitaries,
            budgets=np.arange(min_budget, max_budget),
            load_dir="../../assets/" + gate_set_t,
            save_dir=save_dir + "/" + gate_set_t,
            costs={"T": 1.0},
            num_attempts=num_attempts,
            num_samples=num_samples
        ),
        dict(
            unitaries=unitaries,
            budgets=np.arange(min_budget, max_budget, 0.5),
            load_dir="../../assets/" + gate_set_cost_25,
            save_dir=save_dir + "/" + gate_set_cost_25,
            costs={"T": 1.0, "sqrtT": 2.5},
            num_attempts=num_attempts,
            num_samples=num_samples
        ),
    ]

    for task in tasks:
        run_benchmark(task)
