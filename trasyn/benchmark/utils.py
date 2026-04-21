import re
from pathlib import Path
import numpy as np
import scipy.optimize
from matplotlib.axes import Axes
from tqdm import tqdm
import pickle
from trasyn.synthesis import BudgetPartitioner, Synthesiser, ErgodicPartitioner
from trasyn.utils import random_unitary_2x2
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def fit_raw_data_log(ax: Axes, n: list[int | float], error: NDArray, color: str):
    pairs = np.array(sorted(zip(n, error)))
    n, error = zip(*pairs)
    n = np.array(n)
    error = np.array(error)

    def _func(x, a, b):
        return b + a * x

    popt, pcov = scipy.optimize.curve_fit(_func, n, np.log(1 / error))
    (line,) = plt.plot(
        sorted(n),
        sorted(np.exp(_func(n, *popt))),
        "--",
        label=f"$R = {np.round((1 / popt[0]) * np.log(2), 2)}\log_2(1/\epsilon) - {np.round(popt[1] / popt[0], 2)}$",
        color=color,
        linewidth=2,
    )
    return line


def fit_data_lin(x: list[int | float], y: NDArray, color: str):
    pairs = np.array(sorted(zip(x, y)))
    x, y = zip(*pairs)
    x = np.array(x)
    y = np.array(y)

    def _func(x, a, b):
        return b + a * x

    popt, pcov = scipy.optimize.curve_fit(_func, x, y)
    perr = np.sqrt(np.diag(pcov))
    perr_prop = perr[0]
    plt.plot(
        sorted(x),
        sorted(_func(x, *popt)),
        "--",
        label={
            f"$({np.round(popt[0], 2)} \pm {np.round(perr_prop, 2)})n+{np.round(popt[1], 2)}$"
        },
        color=color,
    )


def fit_data_const(x: list[int | float], y: NDArray, color: str):
    pairs = np.array(sorted(zip(x, y)))
    x, y = zip(*pairs)
    x = np.array(x)
    y = np.array(y)

    def _func(x, b):
        return np.ones(len(x)) * b

    popt, pcov = scipy.optimize.curve_fit(_func, x, y)
    perr = np.sqrt(np.diag(pcov))
    perr_prop = perr[0]
    plt.plot(
        sorted(x),
        sorted(_func(x, popt)),
        "--",
        label={f"${np.round(popt[0], 2)} \pm {np.round(perr_prop, 2)}$"},
        color=color,
    )


def filter_for_zero_error(budget, d_error, error):
    drop_indices = []
    for index, (b, d, e) in enumerate(zip(budget, d_error, error)):
        if np.allclose(d, 0.0) or np.allclose(e, 0.0):
            drop_indices.append(index)

    budget = [val for i, val in enumerate(budget) if i not in drop_indices]
    d_error = [val for i, val in enumerate(d_error) if i not in drop_indices]
    error = [val for i, val in enumerate(error) if i not in drop_indices]

    return np.array(budget), np.array(d_error), np.array(error)


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


def get_median_and_errors(
    data: dict[float | int, list[float]], evaluation_points: list[float | int]
) -> tuple[NDArray, NDArray, list[NDArray]]:
    medians = []
    stds_low = []
    stds_high = []
    evals = []
    for eval in evaluation_points:
        data_eval = data.get(eval, None)
        if data_eval is not None:
            evals.append(eval)
            median, low_err, high_err = get_median_and_skewed_error(data_eval)
            stds_low.append(low_err)
            stds_high.append(high_err)
            medians.append(median)

    return np.array(evals), np.array(medians), [np.array(stds_low), np.array(stds_high)]


def get_median_and_skewed_error(data):
    median = np.median(data)
    low, high = np.percentile(data, [16, 84])
    low_err = median - low
    high_err = high - median

    return median, low_err, high_err


def get_low_err(data):
    median = np.median(data)
    low, _ = np.percentile(data, [16, 84])
    return median - low


def get_high_err(data):
    median = np.median(data)
    _, high = np.percentile(data, [16, 84])
    return high - median


def unfold(data: dict[int | list[float]]):
    key_list = []
    val_list = []
    for key, val in data.items():
        for v in val:
            key_list.append(key)
            val_list.append(v)

    return key_list, val_list


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
    seed: int,
    save_dir: str = "./benchmark_results",
    max_partition_value: int = 5,
    num_attempts: int = 5,
    num_samples: int | None = None,
):
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for nc_budget in tqdm(budgets, desc="nc budget", position=0):
        partitioner = ErgodicPartitioner(
            max_partition_value=max_partition_value,
            total_non_clifford_budget=nc_budget,
            costs=costs,
        )
        syn = Synthesiser(
            partitioner=partitioner,
            load_dir=load_dir,
            num_attempts=num_attempts,
            num_samples=num_samples,
        )

        errors = []
        seqs = []
        for target_unitary in tqdm(
            unitaries, desc="unitaries", leave=False, position=1
        ):
            result = syn.sample_and_synthesize(target_unitary, verbose=False)
            errors.append(result.error)
            seqs.append(result.seqstr)

        # Save per-budget checkpoint
        budget_data = {
            "error_data": errors,
            "seqstr_data": seqs,
            "budgets": budgets,
            "n_unitaries_per_budget": len(unitaries),
            "seed": seed,
        }
        pickle.dump(
            budget_data, open(f"{save_dir}/budget_{nc_budget}_results.pkl", "wb")
        )
