import pickle
import matplotlib.pyplot as plt
from trasyn.results.utils import get_mean_and_std, fit_and_plot, extract_budget_files


dir = "./benchmark_results_100"
colors = ["red", "blue", "green"]
labels = ["T + sqrtT (cost 2)", "T + sqrtT (cost 2.5)", "T"]
gate_sets_dir = [
    "tqshxyz_tequiv_medium_cost_2.5"
]  # "tqshxyz_tequiv_medium",  "tshxyz_tequiv_medium"


for gs, color, label in zip(gate_sets_dir, colors, labels):
    error_data = {}
    sequences = {}
    budgets = []

    load_dir = dir + "/" + gs + "/"
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
