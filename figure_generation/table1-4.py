import time
import numpy as np
from pathlib import Path

from cox_functions import validation_experiment
from wce_functions import validation_wce_experiment


########## Table 1 (Cox, float32)

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


########### Table 2 (Cox, float64)

experiment_name   = "table_2"
dtype             = np.float64

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
print(f"\nTable 2 experiment took {time.time() - t_start:.2f} seconds")


############ Table 3 (WCE, float32)

experiment_name   = "table_3"
n_iteration       = 100
n_patients        = 1000
nknots_list       = [1, 2, 3]
constraint_list   = [None, "right"]
cutoff_list       = [180]
scenario_list     = [
    "exponential_scenario",
    "bi_linear_scenario",
    "early_peak_scenario",
    "inverted_u_scenario",
    "constant_scenario",
    "hat_scenario",
]
dtype             = np.float32

t_start = time.time()
validation_wce_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_iteration=n_iteration,
    n_patients=n_patients,
    max_time=max_time,
    nknots_list=nknots_list,
    constraint_list=constraint_list,
    cutoff_list=cutoff_list,
    hr_candidates=hr_candidates,
    scenario_list=scenario_list,
    dtype=dtype,
    seed=seed,
)
print(f"\nTable 3 experiment took {time.time() - t_start:.2f} seconds")


############# Table 4 (WCE, float64)

experiment_name   = "table_4"
dtype             = np.float64

t_start = time.time()
validation_wce_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_iteration=n_iteration,
    n_patients=n_patients,
    max_time=max_time,
    nknots_list=nknots_list,
    constraint_list=constraint_list,
    cutoff_list=cutoff_list,
    hr_candidates=hr_candidates,
    scenario_list=scenario_list,
    dtype=dtype,
    seed=seed,
)
print(f"\nTable 4 experiment took {time.time() - t_start:.2f} seconds")
