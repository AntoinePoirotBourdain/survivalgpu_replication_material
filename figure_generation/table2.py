import numpy as np
from pathlib import Path

from wce_functions import validation_wce_experiment


######### Validation of wceGPU() against the reference WCE() #########

print("Running WCE validation for table 2...")

experiment_name   = "table_2"
output_folder     = Path("results")
n_iteration       = 100
n_patients        = 1000
max_time          = 365
n_knots_list      = [1, 2, 3]
constraint_list   = [None, "right"]
cutoff_list       = [180]
hr_candidates     = [0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0]
scenario_list     = [
    "exponential_scenario",
    "bi_linear_scenario",
    "early_peak_scenario",
    "inverted_u_scenario",
    "constant_scenario",
    "hat_scenario",
]
dtype             = np.float32
seed              = 42


validation_wce_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_iteration=n_iteration,
    n_patients=n_patients,
    max_time=max_time,
    n_knots_list=n_knots_list,
    constraint_list=constraint_list,
    cutoff_list=cutoff_list,
    hr_candidates=hr_candidates,
    scenario_list=scenario_list,
    dtype=dtype,
    seed=seed,
)
