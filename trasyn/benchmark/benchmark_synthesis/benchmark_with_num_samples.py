import numpy as np
from trasyn.benchmark.utils import generate_unitaries, benchmark_budget, get_suitable_num_samples

SEED = 42
np.random.seed(SEED)

if __name__ == "__main__":
    n_unitaries = 100 # this should be sufficient to represent the distribution
    unitaries = generate_unitaries(n_unitaries)
    min_budget = 1
    max_budget = 22
    num_attempts = 5
    max_partition_value = 10
    save_dir = f"./benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_{num_attempts}_max_partition_value={max_partition_value}_varying_num_samples"
    gate_set_cost_25 = "merged/tqshxyz_tequiv_medium_cost_2.5"
    gate_set_t = "tshxyz_tequiv_medium"

    
    for budget in np.arange(min_budget, max_budget):
        # benchmark_budget(unitaries=unitaries,
        #     budget=budget,
        #     load_dir="../../assets/" + gate_set_t,
        #     save_dir=save_dir + "/" + gate_set_t,
        #     seed=SEED,
        #     costs={"T": 1.0},
        #     num_attempts=num_attempts,
        #     num_samples=get_suitable_num_samples(budget),
        #     max_partition_value=max_partition_value)
        
        benchmark_budget(unitaries=unitaries,
                         budget=budget,
                         load_dir="../../assets/" +  gate_set_cost_25,
                         save_dir=save_dir + "test/" +  gate_set_cost_25,
                         seed=SEED,
                         costs={"T": 1.0, "sqrtT": 2.5},
                         num_attempts=num_attempts,
                         num_samples=2*get_suitable_num_samples(budget),
                         max_partition_value=max_partition_value)
    
