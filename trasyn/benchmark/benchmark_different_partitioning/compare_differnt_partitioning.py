import pickle
import matplotlib.pyplot as plt
from trasyn.benchmark.utils import get_mean_and_std, fit_and_plot, extract_budget_files


dirs = ["./benchmark_results_100_seed_42min", "./benchmark_results_100_seed_42no_min"]
colors = ["red", "blue", "green"]
labels = ["T min", "T no min"]
gate_sets_dir = "tshxyz_tequiv_medium"


for dir, color, label in zip(dirs, colors, labels):
    error_data = {}
    sequences = {}
    budgets = []

    load_dir = dir + "/" + gate_sets_dir + "/"
    for budget, path in extract_budget_files(load_dir):
        with open(path, "rb") as file:
            data = pickle.load(file)

        budgets.append(budget)
        error_data[budget] = data["error_data"]
        sequences[budget] = data["seqstr_data"]

    budgets, error_means, error_stds = get_mean_and_std(error_data, budgets)

    plt.errorbar(
        budgets,
        1 / error_means,
        yerr=error_stds / (error_means**2),
        capsize=5,
        fmt="o-",
        label=label,
        color=color,
        ecolor=color,
    )
    fit_and_plot(budgets, error_means, error_stds, color=color)

plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()