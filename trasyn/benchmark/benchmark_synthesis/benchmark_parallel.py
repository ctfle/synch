import os
from pathlib import Path
import pickle
import cupy as cp
from multiprocessing import Pool

import numpy as np
from numpy._typing import NDArray
from tqdm import tqdm

from trasyn.benchmark.utils import generate_unitaries, get_suitable_num_samples
from trasyn.synthesis import BudgetPartitioner, Sythesiser

SEED = 42
np.random.seed(SEED)


def worker_benchmark(args):
    """Worker for one GPU: (unitaries_slice, nc_budget, other_args..., gpu_id)"""
    (unitaries,
     nc_budget,
     costs,
     load_dir,
     seed,
     save_dir,
     max_partition_value,
     num_attempts,
     num_samples,
     minimize_partitioning,
     gpu_id) = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Isolate GPU
    
    cp.cuda.Device(gpu_id).use()  # Bind CuPy to this GPU

    partitioner = BudgetPartitioner(
        max_partition_value=max_partition_value,
        total_non_clifford_budget=nc_budget,
        costs=costs,
        minimize_partition_count=minimize_partitioning
    )
    syn = Sythesiser(
        partitioner=partitioner,
        load_dir=load_dir,
        num_attempts=num_attempts,
        num_samples=num_samples
    )

    errors = []
    seqs = []
    for target_unitary in tqdm(unitaries, desc=f"GPU {gpu_id} unitaries", leave=False):
        result = syn.sample_and_synthesize(target_unitary, verbose=False)
        errors.append(result.error)
        seqs.append(result.seqstr)

    return errors, seqs, len(unitaries)


def benchmark_unitaries_parallel(
        unitaries: list[NDArray],
        budgets: list[float] | list[int],
        costs: dict[str, float],
        load_dir: str,
        seed: int,
        save_dir: str = "./benchmark_results",
        max_partition_value: int = 5,
        num_attempts: int = 5,
        num_samples: int | None = None,
        minimize_partitioning: bool = True,
        num_gpus: int = 4,  # Target 4 GPUs
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Split unitaries across GPUs
    chunk_size = len(unitaries) // num_gpus
    unitary_chunks = [unitaries[i:i + chunk_size] for i in range(0, len(unitaries), chunk_size)]

    # Prepare args for each GPU worker
    worker_args = [
        (
            unitary_chunks[gpu_id], nc_budget, costs, load_dir, seed, save_dir,
            max_partition_value, num_attempts, num_samples, minimize_partitioning, gpu_id
        )
        for gpu_id in range(num_gpus)
    ]

    # Run parallel across GPUs
    with Pool(processes=num_gpus) as pool:
        results = list(tqdm(pool.imap(worker_benchmark, worker_args), total=num_gpus, desc="GPUs"))

    # Aggregate results
    all_errors = [err for err, _, _ in results]
    all_seqs = [seq for _, seq, _ in results]
    total_n = sum(n for _, _, n in results)

    # Save per-budget checkpoint (flattened)
    budget_data = {
        "error_data": [e for sublist in all_errors for e in sublist],
        "seqstr_data": [s for sublist in all_seqs for s in sublist],
        "budgets": budgets,
        "n_unitaries_per_budget": total_n,
        "seed": seed,
        "num_gpus_used": num_gpus
    }
    pickle.dump(budget_data, open(f"{save_dir}/budget_{nc_budget}_results.pkl", "wb"))


def run_benchmark(args):
    return benchmark_unitaries_parallel(**args)


if __name__ == "__main__":
    n_gpus = cp.cuda.runtime.getDeviceCount()
    n_unitaries = 100
    unitaries = generate_unitaries(n_unitaries)
    min_budget = 17.5
    max_budget = 22
    num_attempts = 5
    save_dir = f"./benchmark_results_{n_unitaries}_seed_{SEED}_num_attempts_{num_attempts}_varying_num_samples"
    gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"  # "tqshxyz_tequiv_short_cost_3"
    gate_set_cost_2 = "tqshxyz_tequiv_medium_cost_2"  # "tqshxyz_tequiv_short"
    gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
    gate_set_t = "tshxyz_tequiv_medium"

    budgets = np.arange(min_budget, max_budget, 0.5)
    for nc_budget in tqdm(budgets, desc="nc budget"):
        tasks = [
            # dict(
            #     unitaries=unitaries,
            #     budgets=np.arange(min_budget, max_budget),
            #     load_dir="../../assets/" + gate_set_t,
            #     save_dir=save_dir + "test/" + gate_set_t,
            #     seed=SEED,
            #     costs={"T": 1.0},
            #     num_attempts=num_attempts,
            #     num_samples=get_suitable_num_samples(budget=nc_budget),
            #     max_partition_value=4,
            #     n_gpus=n_gpus
            # ),
            dict(
                unitaries=unitaries,
                budgets=np.arange(min_budget, max_budget, 0.5),
                load_dir="../../assets/" + gate_set_cost_25,
                save_dir=save_dir + "/" + gate_set_cost_25,
                seed=SEED,
                costs={"T": 1.0, "sqrtT": 2.5},
                num_attempts=num_attempts,
                num_samples=3 * get_suitable_num_samples(budget=nc_budget),
                num_gpus=n_gpus
            ),
        ]

        for task in tasks:
            run_benchmark(task)
# Setup Notes
# Launch on Exoscale gpu3080ti.medium (2x RTX 3080 Ti) or .large (4x) via exo compute instance create ... --template gpu3080ti.medium.
#
# Install CuPy with CUDA support: pip install cupy-cuda12x (match your CUDA version, e.g., 12.x for RTX 30-series).
#
# Test GPU detection first: python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())" should show 4.
#
# Each process gets ~25% of unitaries, speeding up by ~4x for GPU-bound sample_and_synthesize. Monitor with nvidia-smi.
#
# If uneven chunks, adjust padding; for >4 GPUs, scale num_gpus dynamically via cp.cuda.runtime.getDeviceCount().
#
# You reached the Pro search limit on the Free plan. Answers will use basic search until tomorrow.
# Upgrade to Pro for additional usage and access to the top AI models.
