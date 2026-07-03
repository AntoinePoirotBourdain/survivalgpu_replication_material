import time
import numpy as np
from pathlib import Path

from wce_functions import run_wce_experiment


######### Benchmark WCE without bootstraps #########

print("Running WCE benchmark without bootstraps...")

experiment_name   = "figure_5"
output_folder     = Path("benchmark_results")
n_patients_list   = [500, 1000, 5000, 10000, 50000, 100000]
scenario          = "exponential_scenario"
HR_target         = 1.5
max_time          = 365
cutoff            = 180
nknots_list      = [3]
constrained       = None
nbootstraps      = 0
batchsize        = 0
seed              = 42
n_iterations      = 3

# List of (package, dtype, device) couples to benchmark
package_dtype_list = [
    ("survivalgpu", np.float32, "cuda"),
    ("survivalgpu", np.float32, "cpu"),
    ("WCE",         np.float64, "cpu"),
]


t_start = time.time()
run_wce_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    package_dtype_list=package_dtype_list,
    scenario=scenario,
    HR_target=HR_target,
    max_time=max_time,
    cutoff=cutoff,
    nknots_list=nknots_list,
    constrained=constrained,
    nbootstraps=nbootstraps,
    batchsize=batchsize,
    seed=seed,
    n_iterations=n_iterations,
)
print(f"\nFigure 5 experiment took {time.time() - t_start:.2f} seconds")


