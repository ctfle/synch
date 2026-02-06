import numpy as np
from trasyn.benchmark.utils import generate_unitaries, benchmark_on_random_unitaries

SEED = 42
np.random.seed(SEED)
# benchmark the influence of the num_samples chosen


def run_benchmark(args):
    return benchmark_on_random_unitaries(**args)


if __name__ == "__main__":
    n_unitaries = 100
    unitaries = generate_unitaries(n_unitaries)
    min_budget = 20
    max_budget = 21
    num_attempts = 5
    for budget in [14, 16, 20]:
        if budget == 14:
            samples = [20_000]
        else:
            samples = [100, 1000, 10_000, 20_000]
        for num_samples in samples:
        
            save_dir = f"./benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_{num_attempts}_num_samples_{num_samples}"
            gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"  # "tqshxyz_tequiv_short_cost_3"
            gate_set_cost_2 = "tqshxyz_tequiv_medium_cost_2"  # "tqshxyz_tequiv_short"
            gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
            gate_set_t = "tshxyz_tequiv_medium"
        
            tasks = [
                dict(
                    unitaries=unitaries,
                    budgets=np.arange(budget, budget + 1),
                    load_dir="../../assets/" + gate_set_cost_25,
                    save_dir=save_dir + "/" + gate_set_cost_25,
                    seed=SEED,
                    costs={"T": 1.0, "sqrtT": 2.5},
                    num_attempts=num_attempts,
                    num_samples=num_samples
                ),
            ]
    
            for task in tasks:
                run_benchmark(task)
