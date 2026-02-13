import pickle
import matplotlib.pyplot as plt
import numpy as np

from trasyn.benchmark.utils import get_mean_and_std, fit_and_plot, extract_budget_files, \
    fit_and_plot_with_scipy, rescale_costs, merge, unfold, fit_raw_data, get_median_and_errors, \
    get_low_err, get_high_err

dir = "benchmark_results_100_seed_42_num_attempts_5_varying_num_samples"  # "./benchmark_results_1000"
colors = ["red", "blue", "green"]
labels = ["T", "T + sqrtT (cost 2.77)", "T + sqrtT (cost 2)"]
gate_sets_dir = [
    "tshxyz_tequiv_medium", "tqshxyz_tequiv_medium_cost_2.5_large_num_samples" ] # "tqshxyz_tequiv_medium_cost2"


costs = {"t": 1, "q": 2.77}
for gs, color, label in zip(gate_sets_dir, colors, labels):
    error_data = {}
    sequences = {}
    budgets = []

    load_dir = dir + "/" + gs + "/"
    for budget, path in extract_budget_files(load_dir):
        with open(path, "rb") as file:
            data = pickle.load(file)

        error_data = merge(error_data, rescale_costs(data["error_data"],
                                   data["seqstr_data"], budget,
                                   costs=costs))

    if gs == "tqshxyz_tequiv_medium_cost_2.5_large_num_samples":
        budgets, errors = unfold(error_data)
        inv_error = 1/np.array(errors)

        bins = 30  # Number of bins
        counts, bin_edges = np.histogram(budgets, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Get mean/std y-values per bin (ignore empty bins)
        bin_indices = np.digitize(budgets, bin_edges) - 1
        y_binned = np.array([inv_error[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan
                             for i in range(len(bin_edges) - 1)])

        low_err = np.array(
                         [get_low_err(inv_error[bin_indices == i]) if np.any(bin_indices == i) else np.nan
                          for i in range(len(bin_edges) - 1)])

        high_err = np.array(
                         [get_high_err(inv_error[bin_indices == i]) if np.any(bin_indices == i) else np.nan
                          for i in range(len(bin_edges) - 1)])


        plt.errorbar(bin_centers, y_binned,
                     yerr=[low_err, high_err],
                     fmt='o-', capsize=3, label=label)
        fit_raw_data(np.array(budgets), np.array(errors), color=color)

    else:
        budgets, error_means, error_stds = get_median_and_errors(error_data, sorted(list(error_data.keys())))

        plt.errorbar(
            budgets,
            1 / error_means,
            yerr=[error_stds[0] / (error_means**2), error_stds[1] / (error_means**2)],
            capsize=5,
            fmt="o",
            label=label,
            color=color,
            ecolor=color,
        )
        budgets, errors = unfold(error_data)
        fit_raw_data(np.array(budgets), np.array(errors), color=color)
        #plt.scatter(budgets, 1/np.array(errors))
        #fit_and_plot_with_scipy(budgets,error_means,error_stds,color)


plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()
