import pickle
import matplotlib.pyplot as plt
from trasyn.benchmark.utils import get_mean_and_std, extract_budget_files

num_samples = [100, 1000, 10_000, 20_000, 30_000, 40_000]# 100000]
dirs = [f"./benchmark_results_100_seed_42_num_attempts_5_num_samples_{n}" for n in num_samples]
colors = ["red", "blue", "green", "black"]
labels = [f"{n}" for n in num_samples]
gate_sets_dir = "tshxyz_tequiv_medium"

for budget, color in zip([10,14,16,20], colors):
    if budget == 20:
        num_samples = [100, 1000, 10_000, 20_000, 30_000, 40_000]
    else:
        num_samples = [100, 1000, 10_000, 20_000]
    
    error_data = {}
    for dir, n in zip(dirs, num_samples):

        load_dir = dir + "/" + gate_sets_dir + "/"
        for bud, path in extract_budget_files(load_dir):
            if bud == budget:
                with open(path, "rb") as file:
                    data = pickle.load(file)

                error_data[n] = data["error_data"]

    _, error_means, error_stds = get_mean_and_std(error_data, num_samples)
    print(error_means)
    print(num_samples)
    plt.errorbar(
        num_samples,
        1 / error_means,
        yerr=error_stds / (error_means**2),
        capsize=5,
        fmt="o-",
        label=f"nc budget {budget}",
        color=color
        )

plt.ylabel("$(avg. error)^{-1}$")
plt.xlabel("num samples")
plt.legend()
plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()

