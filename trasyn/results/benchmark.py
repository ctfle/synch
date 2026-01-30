from multiprocessing import Pool

from trasyn.synthesis import BudgetPartitioner, Sythesiser
from trasyn.utils import random_unitary_2x2
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np


def benchmark_on_random_unitaries(
    n_unitaries_per_budget: int,
    budgets: list[float] | list[int],
    costs: dict[str, float],
    load_dir: str,
    save_dir: str = "./benchmark_results",
    max_partition_value: int = 5
):
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for nc_budget in tqdm(budgets):
        partitioner = BudgetPartitioner(
            max_partition_value=max_partition_value, total_non_clifford_budget=nc_budget, costs=costs
        )
        syn = Sythesiser(partitioner=partitioner, load_dir=load_dir)

        errors = []
        seqs = []
        for i in range(n_unitaries_per_budget):
            target_unitary = random_unitary_2x2()
            result = syn.sample_and_synthesize(target_unitary, verbose=False)
            errors.append(result.error)
            seqs.append(result.seqstr)

        # Save per-budget checkpoint
        budget_data = {
            "error_data": errors,
            "seqstr_data": seqs,
            "budgets": budgets,
            "n_unitaries_per_budget": n_unitaries_per_budget,
        }
        pickle.dump(
            budget_data, open(f"{save_dir}/budget_{nc_budget}_results.pkl", "wb")
        )


def run_benchmark(args):
    return benchmark_on_random_unitaries(**args)


if __name__ == "__main__":
    n_unitaries = 100
    max_budget = 12
    save_dir = f"./benchmark_results_{n_unitaries}"
    gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"  # "tqshxyz_tequiv_short_cost_3"
    gate_set_cost_2 = "tqshxyz_tequiv_medium"  # "tqshxyz_tequiv_short"
    gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
    gate_set_t = "tshxyz_tequiv_medium"

    tasks = [
        dict(
            n_unitaries_per_budget=n_unitaries,
            budgets=np.arange(2, max_budget, 0.5),
            load_dir="../assets/" + gate_set_cost_25,
            save_dir=save_dir + "/" + gate_set_cost_25,
            costs={"T": 1.0, "sqrtT": 2.5},
        ),
        # dict(
        #     n_unitaries_per_budget=n_unitaries,
        #     budgets=np.arange(2, max_budget),
        #     gate_set=gate_set_t,
        #     save_dir=save_dir,
        #     costs={"T": 1.0},
        # ),
        # dict(
        #     n_unitaries_per_budget=n_unitaries,
        #     budgets=np.arange(2, max_budget),
        #     gate_set=gate_set_cost_2,
        #     save_dir=save_dir,
        #     costs={"T": 1.0, "sqrtT": 2.0},
        # ),
    ]

    with Pool(processes=3) as pool:  # or processes=None for "cpu_count()"
        pool.map(run_benchmark, tasks)
