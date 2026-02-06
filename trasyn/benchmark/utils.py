import re
from pathlib import Path
import numpy as np
import scipy.optimize
from scipy.constants import sigma
from tqdm import tqdm
import pickle
from trasyn.synthesis import BudgetPartitioner, Sythesiser
from trasyn.utils import random_unitary_2x2

SEED = 42
np.random.seed(SEED)
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def fit_and_plot(n: list[int | float], error: NDArray, d_error: NDArray, color: str):
    # Transform to log(1/ε)
    propagated_error = 1 / error * d_error
    # Fit linear regression
    params = np.polyfit(np.log(1 / error), n, 1, w=1/ propagated_error)
    # Plot fit line
    n_fit = np.linspace(min(n), max(n), 100)
    plt.plot(
        n_fit,
        np.exp(1 / params[0] * n_fit - params[1] / params[0]),
        "--",
        color=color,
        label=f"fit: n =  ${np.round(params[0], 2)} \\log(1/\\epsilon)$",
    )


def fit_and_plot_with_scipy(n: list[int | float], error: NDArray, d_error: NDArray, color: str):
    def _func(x, a, b):
        return b * np.exp(a*x)

    popt, pcov = scipy.optimize.curve_fit(_func, n, 1/error, sigma=(1/error**2 *d_error))
    plt.plot(n, _func(n, *popt), '--',
             label=f'$n = {np.round(1/popt[0],2)}\log(1/\epsilon)$',
             color=color)



def get_mean_and_std(
    data: dict[float | int, list[float]], evaluation_points: list[float | int]
) -> tuple[NDArray, NDArray, NDArray]:
    means = []
    stds = []
    evals = []
    for eval in evaluation_points:
        data_eval = data.get(eval, None)
        if data_eval is not None:
            evals.append(eval)
            stds.append(np.std(data_eval))
            means.append(np.mean(data_eval))

    return np.array(evals), np.array(means), np.array(stds)


def analyse_sequence(seqs: list[str], gates: list[str]) -> list[list[int]]:
    """
    Counts the numbers of t and q in each sequence.
    """
    return [[s.count(g) for g in gates] for s in seqs]


def extract_budget_files(directory: str) -> list[tuple[float, Path]]:
    path = Path(directory)
    pattern = re.compile(r"budget_(\d+(?:\.\d+)?)")
    results: list[tuple[float, Path]] = []

    for entry in path.iterdir():
        if not entry.is_file():
            continue
        matches = pattern.findall(entry.name)
        for num_str in matches:
            results.append((float(num_str), entry))

    return sorted(results, key=lambda x: x[0])



def generate_unitaries(num_unitaries: int) -> list[NDArray]:
    return [random_unitary_2x2() for _ in range(num_unitaries)]


def benchmark_on_random_unitaries(
    unitaries: list[NDArray],
    budgets: list[float] | list[int],
    costs: dict[str, float],
    load_dir: str,
    save_dir: str = "./benchmark_results",
    max_partition_value: int = 5,
    num_attempts: int = 5,
    num_samples: int | None = None,
    minimize_partitioning: bool = True
):
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for nc_budget in tqdm(budgets, desc="nc budget", position=0):
        partitioner = BudgetPartitioner(
            max_partition_value=max_partition_value, total_non_clifford_budget=nc_budget, costs=costs,
            minimize_partition_count=minimize_partitioning
        )
        syn = Sythesiser(partitioner=partitioner,
                         load_dir=load_dir,
                         num_attempts= num_attempts,
                         num_samples=num_samples)

        errors = []
        seqs = []
        for target_unitary in tqdm(unitaries, desc="unitaries", leave=False, position=1):
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


def benchmark_budget(
    unitaries: list[NDArray],
    budget: int | float,
    costs: dict[str, float],
    load_dir: str,
    save_dir: str = "./benchmark_results",
    max_partition_value: int = 5,
    num_attempts: int = 5,
    num_samples: int | None = None,
    minimize_partitioning: bool = True
):
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    partitioner = BudgetPartitioner(
        max_partition_value=max_partition_value, total_non_clifford_budget=budget, costs=costs,
        minimize_partition_count=minimize_partitioning
    )
    syn = Sythesiser(partitioner=partitioner,
                     load_dir=load_dir,
                     num_attempts= num_attempts,
                     num_samples=num_samples)

    errors = []
    seqs = []
    for target_unitary in tqdm(unitaries, desc="unitaries", leave=False, position=1):
        result = syn.sample_and_synthesize(target_unitary, verbose=False)
        errors.append(result.error)
        seqs.append(result.seqstr)

    # Save per-budget checkpoint
    budget_data = {
        "error_data": errors,
        "seqstr_data": seqs,
        "n_unitaries_per_budget": len(unitaries),
        "seed": SEED
    }
    pickle.dump(
        budget_data, open(f"{save_dir}/budget_{budget}_num_samples_{num_samples}_results.pkl", "wb")
    )


def get_suitable_num_samples(budget: int):
    """ These number of samples are obtained by empirically checking convergence
    of the corresponding errors. """
    if budget <= 10:
        num_samples = 1000
    elif budget > 10 and budget < 17:
        num_samples = 10_000
    else:
        num_samples = 20_000 + 10_000 * (budget - 17)

    return num_samples
