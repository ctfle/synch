import numpy as np
from trasyn.benchmark.utils import generate_unitaries, benchmark_budget, SEED, get_suitable_num_samples

if __name__ == "__main__":
    n_unitaries = 100 # this should be sufficient to represent the distribution
    unitaries = generate_unitaries(n_unitaries)
    min_budget = 2
    max_budget = 20
    num_attempts = 5
    save_dir = f"./benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_{num_attempts}_varying_num_samples"
    gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"  # "tqshxyz_tequiv_short_cost_3"
    gate_set_cost_2 = "tqshxyz_tequiv_medium_cost_2"  # "tqshxyz_tequiv_short"
    gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
    gate_set_t = "tshxyz_tequiv_medium"

    
    for budget in np.arange(min_budget, max_budget):
        benchmark_budget(unitaries=unitaries,
            budget=budget,
            load_dir="../../assets/" + gate_set_t,
            save_dir=save_dir + "/" + gate_set_t,
            costs={"T": 1.0},
            num_attempts=num_attempts,
            num_samples=get_suitable_num_samples(budget))

    # benchmark_budget(unitaries=unitaries,
    #                  budget=budget,
    #                  load_dir="../../assets/" + gate_set_t,
    #                  save_dir=save_dir + "/" + gate_set_t,
    #                  costs={"T": 1.0, "sqrtT": 2.5},
    #                  num_attempts=num_attempts,
    #                  num_samples=get_suitable_num_samples(budget))

