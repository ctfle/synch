import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def fit_and_plot(n: list, error: NDArray, d_error: NDArray,  color: str):
    # Transform to log(1/ε)
    log_inv_eps = np.log(1 / error)  # log(1/ε)
    propagated_error = 1/error * d_error
    # Fit linear regression
    params = np.polyfit(np.log(1 / error), n, 1, w=1/propagated_error)
    # Plot fit line
    n_fit = np.linspace(min(n), max(n), 100)
    plt.plot(n_fit, np.exp(1 / params[0] * n_fit - params[1] / params[0]),"--", color=color,
             label=f"fit: n =  ${np.round(params[0], 2)} \\log(1/\\epsilon)$")


def get_mean_and_std(data: dict[float | int, list[float]], 
                     evaluation_points: list[float | int]) -> tuple[NDArray, NDArray, NDArray]:
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



dir = "./benchmark_results_5000"
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