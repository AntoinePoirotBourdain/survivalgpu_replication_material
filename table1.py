import sys
import os
import time
import numpy as np
from pathlib import Path

lib_path = os.path.abspath("/home/dev/survivalGPU/python")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from cox_functions import validation_experiment


######### Validation of coxphGPU() against the reference coxph() #########

print("Running Cox validation for table 1...")

experiment_name   = "table_1"
output_folder     = Path("results")
n_patients_list   = [1000]
covariate_combination = [
    (1, 0),  # 1 constant covariate
    (3, 0),  # 3 constant covariates
    (0, 1),  # 1 time-dependent covariate
    (0, 3),  # 3 time-dependent covariates
]
hr_candidates     = [0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0]
beta_candidates   = [np.log(hr) for hr in hr_candidates]
n_simulations     = 100
ties_list         = ("efron", "breslow")
max_time          = 365
seed              = 42
dtype             = np.float32


t_start = time.time()
validation_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    covariate_combination=covariate_combination,
    beta_candidates=beta_candidates,
    n_simulations=n_simulations,
    ties_list=ties_list,
    max_time=max_time,
    seed=seed,
    dtype=dtype,
)
print(f"\nTable 1 experiment took {time.time() - t_start:.2f} seconds")
