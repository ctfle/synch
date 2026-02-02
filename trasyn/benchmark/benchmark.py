from multiprocessing import Pool
import numpy as np
SEED = 42
np.random.seed(SEED)
from trasyn.synthesis import BudgetPartitioner, Sythesiser
from numpy.typing import NDArray
from trasyn.utils import random_unitary_2x2
from tqdm import tqdm
from pathlib import Path
import pickle


def generate_unitaries(num_unitaries: int) -> list[NDArray]:
    return [random_unitary_2x2() for _ in range(num_unitaries)]


def benchmark_on_random_unitaries(
    unitaries: list[NDArray],
    budgets: list[float] | list[int],
    costs: dict[str, float],
    load_dir: str,
    save_dir: str = "./benchmark_results",
    max_partition_value: int = 5,
    num_attempts: int = 10
):
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for nc_budget in tqdm(budgets):
        partitioner = BudgetPartitioner(
            max_partition_value=max_partition_value, total_non_clifford_budget=nc_budget, costs=costs
        )
        syn = Sythesiser(partitioner=partitioner, load_dir=load_dir,num_attempts= num_attempts)

        errors = []
        seqs = []
        for target_unitary in unitaries:
            result = syn.sample_and_synthesize(target_unitary, verbose=False)
            errors.append(result.error)
            seqs.append(result.seqstr)

        # Save per-budget checkpoint
        budget_data = {
            "error_data": errors,
            "seqstr_data": seqs,
            "budgets": budgets,
            "n_unitaries_per_budget": len(unitaries),
            "seed": SEED
        }
        pickle.dump(
            budget_data, open(f"{save_dir}/budget_{nc_budget}_results.pkl", "wb")
        )


def run_benchmark(args):
    return benchmark_on_random_unitaries(**args)


if __name__ == "__main__":
    n_unitaries = 100
    unitaries = generate_unitaries(n_unitaries)
    min_budget = 2
    max_budget = 16
    save_dir = f"./benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_10"
    gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"  # "tqshxyz_tequiv_short_cost_3"
    gate_set_cost_2 = "tqshxyz_tequiv_medium_cost_2"  # "tqshxyz_tequiv_short"
    gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
    gate_set_t = "tshxyz_tequiv_medium"

    tasks = [
        dict(
            unitaries=unitaries,
            budgets=np.arange(min_budget, max_budget),
            load_dir="../assets/" + gate_set_t,
            save_dir=save_dir + "/" + gate_set_t,
            costs={"T": 1.0},
        ),
        dict(
            unitaries=unitaries,
            budgets=np.arange(min_budget, max_budget, 0.5),
            load_dir="../assets/" + gate_set_cost_25,
            save_dir=save_dir + "/" + gate_set_cost_25,
            costs={"T": 1.0, "sqrtT": 2.5},
        ),
        # dict(
        #     n_unitaries_per_budget=n_unitaries,
        #     budgets=np.arange(2, max_budget),
        #     gate_set=gate_set_cost_2,
        #     save_dir=save_dir,
        #     costs={"T": 1.0, "sqrtT": 2.0},
        # ),
    ]


    # run in parallel
    # with Pool(processes=3) as pool:
    #     pool.map(run_benchmark, tasks)

    for task in tasks:
        run_benchmark(task)
