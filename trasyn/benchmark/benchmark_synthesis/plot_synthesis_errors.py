import pickle
import matplotlib.pyplot as plt
from trasyn.benchmark.utils import get_mean_and_std, fit_and_plot, extract_budget_files, fit_and_plot_with_scipy


dir = "benchmark_results_100_seed_42_num_attempts_5"# "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples"  #
colors = ["red", "blue", "green", "orange", "black", "yellow"]
labels = ["T", "T + sqrtT (cost 2.5)", "T + sqrtT (cost 2.5)", "T + sqrtT (cost 2.5) new partitoner", "T + sqrtT (cost 2.5) new partitoner gpu", "test"]
gate_sets_dir = ["_max_partition_value=10_varying_num_samples/tshxyz_tequiv_large", 
                 #"_max_partition_value=10_varying_num_samples/merged/tqshxyz_tequiv_large_cost_2.5_all", 
                 #"_max_partition_value=8_varying_num_samplestest/merged/tqshxyz_tequiv_large_cost_2.5_all",
                 #"_max_partition_value=8_varying_num_samples_new_partitioner/merged/tqshxyz_tequiv_large_cost_2.5_all",
                 #"_max_partition_value=8_varying_num_samples_new_ergodic_partitioner/merged/tqshxyz_tequiv_large_cost_2.5_all",
                 "_max_partition_value=8_test_new_ergodic_partitioner/merged/tqshxyz_tequiv_large_cost_2.5_all"
                 ]
                 #"_max_partition_value=8_varying_num_samples_new_partitioner_1/merged/tqshxyz_tequiv_large_cost_2.5_all"]
    #["tshxyz_tequiv_medium", "tqshxyz_tequiv_medium_cost_2.5_large_num_samples" ] # "tqshxyz_tequiv_medium_cost2"

for gs, color, label in zip(gate_sets_dir, colors, labels):
    error_data = {}
    sequences = {}
    budgets = []

    load_dir = dir + gs + "/"
    for budget, path in extract_budget_files(load_dir):
        with open(path, "rb") as file:
            data = pickle.load(file)

        budgets.append(budget)
        error_data[budget] = data["error_data"]
        sequences[budget] = data["seqstr_data"]
    print(budgets)
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
    if gs == "merged/tqshxyz_tequiv_large_cost_2.5_all":
        print(budgets,budgets[:18])
        fit_and_plot_with_scipy(budgets[:18], error_means[:18], error_stds[:18], color=color)

    else:
        fit_and_plot_with_scipy(budgets, error_means, error_stds, color=color)

plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()


dirs = [ "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples/tqshxyz_tequiv_medium_cost_2.5",
"benchmark_results_100_seed_42_num_attempts_5_max_partition_value=10_varying_num_samples/merged/tqshxyz_tequiv_large_cost_2.5_all",
        "benchmark_results_100_seed_42_num_attempts_5_max_partition_value=8_varying_num_samples_new_partitioner/merged/tqshxyz_tequiv_large_cost_2.5_all",
         "benchmark_results_100_seed_42_num_attempts_5_max_partition_value=8_varying_num_samples_new_ergodic_partitioner/merged/tqshxyz_tequiv_large_cost_2.5_all",
        "benchmark_results_100_seed_42_num_attempts_5_max_partition_value=8_test_new_ergodic_partitioner/merged/tqshxyz_tequiv_large_cost_2.5_all"
         ] # "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples"  #
colors = ["red", "blue", "green", "orange", "black", "yellow"]
labels = [ "T + sqrtT (cost 2.5) original", "T + sqrtT (cost 2.5) increased max partition", "T + sqrtT (cost 2.5) including permutations", "T + sqrtT (cost 2.5) ermutations + better partitioning", "T + sqrtT (cost 2.5) ergodic"]
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



dirs = ["benchmark_results_100_seed_42_num_attempts_5_max_partition_value=10_varying_num_samples/tshxyz_tequiv_large",
        "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples/tshxyz_tequiv_medium"] # "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples"  #
colors = ["red", "blue", "green"]
labels = ["T increased max partition", "T"]
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