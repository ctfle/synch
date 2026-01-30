import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def fit_and_plot(n: list, error: NDArray, d_error: NDArray, color: str):
    # Transform to log(1/ε)
    log_inv_eps = np.log(1 / error)  # log(1/ε)
    propagated_error = 1 / error * d_error
    # Fit linear regression
    params = np.polyfit(np.log(1 / error), n, 1, w=1 / propagated_error)
    # Plot fit line
    n_fit = np.linspace(min(n), max(n), 100)
    plt.plot(
        n_fit,
        np.exp(1 / params[0] * n_fit - params[1] / params[0]),
        "--",
        color=color,
        label=f"fit: n =  ${np.round(params[0], 2)} \\log(1/\\epsilon)$",
    )


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
