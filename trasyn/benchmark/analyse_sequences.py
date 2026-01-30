import pickle
import matplotlib.pyplot as plt
import numpy as np
from trasyn.benchmark.utils import (
    get_mean_and_std,
    fit_and_plot,
    analyse_sequence,
    extract_budget_files,
)


dir = "./benchmark_results_100"
colors = ["red", "blue", "green"]
labels = ["T + sqrtT (cost 2)", "T + sqrtT (cost 2.5)", "T"]
gate_sets = ["tqshxyz_tequiv_medium_cost_2.5", "tshxyz_tequiv_medium"]
# gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"
# gate_set_cost_2 = "tqshxyz_tequiv_medium"
# gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
# gate_set_t = "tshxyz_tequiv_medium"


# get the t and q count for each budget point
# cost 2.5
error_data = {}
sequences = {}
budgets = []

load_dir = dir + "/" + gate_sets[0] + "/"
for budget, path in extract_budget_files(load_dir):
    with open(path, "rb") as file:
        data = pickle.load(file)

    budgets.append(budget)
    error_data[budget] = data["error_data"]
    sequences[budget] = data["seqstr_data"]


gates = ["t", "q"]
budget_points = []
gates_means = [[] for gate in gates]
gates_std = [[] for gate in gates]
for budget, sequence_data in sequences.items():
    budget_points.append(budget)
    analysed_seqs = analyse_sequence(sequence_data, gates)
    for g, gate in enumerate(gates):
        gate_occurences = list(map(lambda x: x[g], analysed_seqs))
        gates_means[g].append(np.mean(gate_occurences))
        gates_std[g].append(np.std(gate_occurences))

for g, gate in enumerate(gates):
    plt.errorbar(
        budget_points,
        gates_means[g],
        yerr=gates_std[g],
        capsize=5,
        fmt="o-",
        label=gate,
        color=colors[g],
        ecolor=colors[g],
    )

plt.ylabel("(avg. gate occurrence)")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()


# compute true total cost for the found sequences
gates = ["t", "q"]
costs = [1, 2.5]
budget_points = []
cost_means = []
cost_std = []
for budget, sequence_data in sequences.items():
    budget_points.append(budget)
    analysed_seqs = analyse_sequence(sequence_data, gates)
    total_cost = list(
        map(lambda x: sum(costs[g] * x[g] for g in range(len(x))), analysed_seqs)
    )
    cost_means.append(np.mean(total_cost))
    cost_std.append(np.std(total_cost))

plt.errorbar(
    budget_points, cost_means, yerr=cost_std, capsize=5, fmt="o-", label="costs"
)

plt.ylabel("(avg. gate occurrence)")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()


def get_ratio(x):
    if x[0] != 0:
        return x[1] / x[0]
    else:
        return 10


gates = ["t", "q"]
budget_points = []
ratio_means = []
ratio_std = []
for budget, sequence_data in sequences.items():
    budget_points.append(budget)
    analysed_seqs = analyse_sequence(sequence_data, gates)
    ratio = list(map(get_ratio, analysed_seqs))
    ratio_means.append(np.mean(ratio))
    ratio_std.append(np.std(ratio))

for g, gate in enumerate(gates):
    plt.errorbar(
        budget_points,
        ratio_means,
        yerr=ratio_std,
        capsize=5,
        fmt="o-",
        label=gate,
        color=colors[g],
        ecolor=colors[g],
    )

plt.ylabel("(avg. gate occurrence)")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()
