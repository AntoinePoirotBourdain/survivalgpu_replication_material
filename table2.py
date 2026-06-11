import sys
import os
import numpy as np
from pathlib import Path

lib_path = os.path.abspath("/home/dev/survivalGPU/python")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

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
HR_list           = [1.3]
scenario_list     = ["exponential_scenario"]
dtype             = np.float32


validation_wce_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_iteration=n_iteration,
    n_patients=n_patients,
    max_time=max_time,
    n_knots_list=n_knots_list,
    constraint_list=constraint_list,
    cutoff_list=cutoff_list,
    HR_list=HR_list,
    scenario_list=scenario_list,
    dtype=dtype,
)
