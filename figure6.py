import sys
import os
import time
import numpy as np
from pathlib import Path

lib_path = os.path.abspath("/home/dev/survivalGPU/python")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from wce_functions import run_wce_experiment


######### Benchmark WCE with 500 bootstraps #########

experiment_name   = "figure_6"
output_folder     = Path("results/")
n_patients_list   = [500, 1000, 5000, 10000, 50000, 100000]
scenario          = "exponential_scenario"
HR_target         = 1.3
max_time          = 365
cutoff            = 180
n_knots_list      = [3]
constrained       = None
n_bootstraps      = 500
batch_size        = 2
seed              = 42
n_iterations      = 1

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
    n_knots_list=n_knots_list,
    constrained=constrained,
    n_bootstraps=n_bootstraps,
    batch_size=batch_size,
    seed=seed,
    n_iterations=n_iterations,
)
print(f"\nFigure 6 experiment took {time.time() - t_start:.2f} seconds")


