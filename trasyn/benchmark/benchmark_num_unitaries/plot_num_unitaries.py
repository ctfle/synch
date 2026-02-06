import pickle
import matplotlib.pyplot as plt
from trasyn.benchmark.utils import get_mean_and_std, extract_budget_files

num_unitaries = [10, 50, 100, 200, 500, 1000]# 100000]
dirs = [f"./benchmark_results_{n}_seed_42_num_attempts_5_num_samples_1000" for n in num_unitaries]
colors = ["red", "blue", "green", "black"]
labels = [f"{n}" for n in num_unitaries]
gate_sets_dir = "tshxyz_tequiv_medium"

error_data = {}
for dir, n in zip(dirs, num_unitaries):

    load_dir = dir + "/" + gate_sets_dir + "/"
    for bud, path in extract_budget_files(load_dir):

        with open(path, "rb") as file:
            data = pickle.load(file)

        error_data[n] = data["error_data"]

_, error_means, error_stds = get_mean_and_std(error_data, num_unitaries)
print(error_means)
plt.errorbar(
        num_unitaries,
        1 / error_means,
        yerr=error_stds / (error_means**2),
        capsize=5,
        fmt="o-"
    )

plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("num unitaries")
plt.legend()
plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()

