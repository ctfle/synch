import pickle
import matplotlib.pyplot as plt

import numpy as np

from trasyn.src.benchmark.utils import (
    extract_budget_files,
    unfold,
    get_median_and_errors,
    get_low_err,
    get_high_err,
    fit_raw_data_log,
)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 28
fig, ax = plt.subplots(figsize=(16, 9))
# with plt.style.context('default'):
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.legend(loc="best", prop={"size": 28})
ax.yaxis.get_offset_text().set_fontsize(28)

# Adjust box thickness
for spine in ax.spines.values():
    spine.set_linewidth(3)
# Thicker  ticks
ax.tick_params(axis="both", which="major", width=3, length=12)  # length is optional
ax.tick_params(axis="both", which="minor", width=1.5, length=8)  # Adjust as needed
ax.tick_params(axis="both", labelsize=28)


colors = ["red", "blue", "green"]
labels = ["$C_2$", "$C_3$", "T + sqrtT (cost 2)"]

gate_sets_dir = [
    "benchmark_results/tshxyz",
    "benchmark_results/tqshxyz",
]

costs = {"t": 1, "q": 2.5}
fit_lines = []
error_plots = []
error_labels = []

for gs, color, label in zip(gate_sets_dir, colors, labels):
    error_data = {}
    sequences = {}
    budgets = []

    load_dir = gs + "/"
    for budget, path in extract_budget_files(load_dir):
        with open(path, "rb") as file:
            data = pickle.load(file)

        error_data[budget] = data["error_data"]

    if gs == "":
        budgets, errors = unfold(error_data)
        inv_error = 1 / np.array(errors)

        bins = 28  # Number of bins
        counts, bin_edges = np.histogram(budgets, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Get mean/std y-values per bin (ignore empty bins)
        bin_indices = np.digitize(budgets, bin_edges) - 1
        y_binned = np.array(
            [
                inv_error[bin_indices == i].mean()
                if np.any(bin_indices == i)
                else np.nan
                for i in range(len(bin_edges) - 1)
            ]
        )

        low_err = np.array(
            [
                get_low_err(inv_error[bin_indices == i])
                if np.any(bin_indices == i)
                else np.nan
                for i in range(len(bin_edges) - 1)
            ]
        )

        high_err = np.array(
            [
                get_high_err(inv_error[bin_indices == i])
                if np.any(bin_indices == i)
                else np.nan
                for i in range(len(bin_edges) - 1)
            ]
        )

        plt.errorbar(
            bin_centers,
            y_binned,
            yerr=[low_err, high_err],
            fmt="o",
            capsize=5,
            label=label,
            color=color,
            markersize=10,
        )
        fit_raw_data_log(ax, np.array(budgets), np.array(errors), color=color)

    else:
        budgets, error_means, error_stds = get_median_and_errors(
            error_data, sorted(list(error_data.keys()))
        )
        errs = ax.errorbar(
            budgets,
            1 / error_means,
            yerr=[error_stds[0] / (error_means**2), error_stds[1] / (error_means**2)],
            capsize=7,
            fmt="o",
            label=label,
            color=color,
            ecolor=color,
            markersize=7,
        )
        budgets, errors = unfold(error_data)
        line = fit_raw_data_log(ax, np.array(budgets), np.array(errors), color=color)
        fit_lines.append(line)
        error_plots.append(errs[0])
        error_labels.append(label)


legend = ax.legend(handles=fit_lines, loc="lower right")
ax.add_artist(legend)
ax.legend(handles=error_plots, labels=error_labels, loc="upper left")
plt.ylabel("$\epsilon^{-1}$")
plt.xlabel("$R$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
