import pickle
import matplotlib.pyplot as plt
import numpy as np

gate_set_cost_3 = "tqshxyz_tequiv_short_cost_3"
gate_set = "tqshxyz_tequiv_short"

with open(f"./benchmark_results_500/{gate_set_cost_3}_results.pkl", "rb") as file:
    data = pickle.load(file)

print(data)
lower = data["config"]["nc_budget_lower"]
upper = data["config"]["nc_budget_upper"]
plt.errorbar(np.arange(lower, upper) ,data["error_data_sqrt_t"], yerr=data["std_sqrt_t"], capsize=5, fmt='o-',label="T + sqrtT (cost 3)")


with open(f"./benchmark_results_500/{gate_set}_results.pkl", "rb") as file:
    data = pickle.load(file)

lower = data["config"]["nc_budget_lower"]
upper = data["config"]["nc_budget_upper"]
plt.errorbar(np.arange(lower, upper), data["error_data_sqrt_t"], yerr=data["std_sqrt_t"], capsize=5, fmt='o-',label="T + sqrtT (cost 2)")
plt.errorbar(np.arange(lower, upper), data["error_data_t"], yerr=data["std_t"], capsize=5, fmt='o-', label="T")
plt.ylabel("avg synthesis error")
plt.xlabel("non-clifford budget")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()