import pickle
import matplotlib.pyplot as plt
from trasyn.results.utils import get_mean_and_std, fit_and_plot


dir = "../benchmark_results_500"
colors = ["red", "blue", "green"]
labels = ["T + sqrtT (cost 2)", "T + sqrtT (cost 2.5)", "T"]
gate_sets = ["tqshxyz_tequiv_medium", "tqshxyz_tequiv_medium_cost_2.5", "tshxyz_tequiv_medium"]
# gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"
# gate_set_cost_2 = "tqshxyz_tequiv_medium"
# gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
# gate_set_t = "tshxyz_tequiv_medium"


for gs, color, label in zip(gate_sets, colors, labels):
    with open(dir + f"/{gs}_results.pkl", "rb") as file:
        data = pickle.load(file)
    
    budgets = data["budgets"]
    error_data = data["error_data"]
    sequences = data["seqstr_data"]
    budgets, error_means, error_stds =  get_mean_and_std(error_data, budgets)
    
    plt.errorbar(budgets, 1/error_means, yerr=error_stds /(error_means **2), capsize=5, fmt='o-',label=label, color=color, ecolor=color)
    fit_and_plot(budgets, error_means, error_stds ,color=color)

plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()