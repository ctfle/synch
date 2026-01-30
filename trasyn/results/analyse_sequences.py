import pickle
import matplotlib.pyplot as plt
import numpy as np
from trasyn.results.utils import get_mean_and_std, fit_and_plot, analyse_sequence


dir = "../benchmark_results_500"
colors = ["red", "blue", "green"]
labels = ["T + sqrtT (cost 2)", "T + sqrtT (cost 2.5)", "T"]
gate_sets = ["tqshxyz_tequiv_medium_cost_2.5", "tshxyz_tequiv_medium"]
# gate_set_cost_3 = "tqshxyz_tequiv_medium_cost_3"
# gate_set_cost_2 = "tqshxyz_tequiv_medium"
# gate_set_cost_25 = "tqshxyz_tequiv_medium_cost_2.5"
# gate_set_t = "tshxyz_tequiv_medium"


# get the t and q count for each budget point
# cost 2.5
with open(dir + f"/{gate_sets[0]}_results.pkl", "rb") as file:
    data = pickle.load(file)

budgets = data["budgets"]
sequences = data["seqstr_data"]

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
    plt.errorbar(budget_points, gates_means[g], yerr=gates_std[g], capsize=5, fmt='o-',label=gate, color=colors[g], ecolor=colors[g])

plt.ylabel("(avg. gate occurrence)")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()

