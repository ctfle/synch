import pickle
import matplotlib.pyplot as plt
from trasyn.benchmark.utils import get_mean_and_std, fit_and_plot, extract_budget_files, fit_and_plot_with_scipy


dir = "benchmark_results_100_seed_42_num_attempts_5_num_samples_1000_max_partition_value=10"# "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples"  #
colors = ["red", "blue", "green"]
labels = ["T", "T + sqrtT (cost 2.5)", "T + sqrtT (cost 2)"]
gate_sets_dir = ["tshxyz_tequiv_large", "merged/tqshxyz_tequiv_large_cost_2.5_all" ]
    #["tshxyz_tequiv_medium", "tqshxyz_tequiv_medium_cost_2.5_large_num_samples" ] # "tqshxyz_tequiv_medium_cost2"

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

    fit_and_plot_with_scipy(budgets, error_means, error_stds, color=color)

plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()


dirs = ["benchmark_results_100_seed_42_num_attempts_5_num_samples_1000_max_partition_value=10/merged/tqshxyz_tequiv_large_cost_2.5_all", 
        "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples/tqshxyz_tequiv_medium_cost_2.5"] # "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples"  #
colors = ["red", "blue", "green"]
labels = ["T + sqrtT (cost 2.5) new", "T + sqrtT (cost 2.5) old"]
    #["tshxyz_tequiv_medium", "tqshxyz_tequiv_medium_cost_2.5_large_num_samples" ] # "tqshxyz_tequiv_medium_cost2"

for dir, color, label in zip(dirs, colors, labels):
    error_data = {}
    sequences = {}
    budgets = []

    load_dir = dir
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

    fit_and_plot_with_scipy(budgets, error_means, error_stds, color=color)

plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()



dirs = ["benchmark_results_100_seed_42_num_attempts_5_num_samples_1000_max_partition_value=10/tshxyz_tequiv_large", 
        "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples/tshxyz_tequiv_medium"] # "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples"  #
colors = ["red", "blue", "green"]
labels = ["T new", "T old"]
    #["tshxyz_tequiv_medium", "tqshxyz_tequiv_medium_cost_2.5_large_num_samples" ] # "tqshxyz_tequiv_medium_cost2"

for dir, color, label in zip(dirs, colors, labels):
    error_data = {}
    sequences = {}
    budgets = []

    load_dir = dir
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

    fit_and_plot_with_scipy(budgets, error_means, error_stds, color=color)

plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()