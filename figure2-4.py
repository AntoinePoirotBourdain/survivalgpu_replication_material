import sys
import os
import numpy as np
from pathlib import Path

lib_path = os.path.abspath("/home/dev/survivalGPU/python")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from cox_functions import run_cox_experiment


beta_candidates = [np.log(0.25), np.log(0.5), np.log(0.75), np.log(1.5), np.log(2.0), np.log(3.0), np.log(4.0)]


########## Figure 2

experiment_name = "figure_2"
n_patients_list = [250, 500]
n_constant_covariates = 1
n_time_dependent_covariates = 0
n_bootstraps = 0
output_folder = Path("results/")
n_iterations = 3

package_device_dtype_list = [
    ("survivalgpu", "cuda", np.float32),
    ("survivalgpu", "cpu", np.float32),
    ("survival", "cpu", np.float64),
    ("lifelines", "cpu", np.float64),
]

run_cox_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    covariate_combination=[(n_constant_covariates, n_time_dependent_covariates)],
    package_device_dtype_list=package_device_dtype_list,
    beta_candidates=beta_candidates,
    n_bootstraps=n_bootstraps,
    n_iterations=n_iterations,
)


########### Figure 3

experiment_name = "figure_3"
n_patients_list = [250, 500]
n_constant_covariates = 1
n_time_dependent_covariates = 0
n_bootstraps = 1000
output_folder = Path("results/")
n_iterations = 3

package_device_dtype_list = [
    ("survivalgpu", "cuda", np.float32),
    ("survivalgpu", "cpu", np.float32),
    ("survival", "cpu", np.float64),
    ("lifelines", "cpu", np.float64),
]

run_cox_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    covariate_combination=[(n_constant_covariates, n_time_dependent_covariates)],
    package_device_dtype_list=package_device_dtype_list,
    beta_candidates=beta_candidates,
    n_bootstraps=n_bootstraps,
    n_iterations=n_iterations,
)


############ Figure 4

experiment_name = "figure_4"
n_patients_list = [250, 500]
n_constant_covariates = 1
n_time_dependent_covariates = 0
n_bootstraps = 0
output_folder = Path("results/")
n_iterations = 3

package_device_dtype_list = [
    ("survivalgpu", "cuda", np.float32),
    ("survival", "cpu", np.float64),
    ("xgboost", "cpu", np.float32),
    ("torchsurv", "cpu", np.float32),
]

run_cox_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    covariate_combination=[(n_constant_covariates, n_time_dependent_covariates)],
    package_device_dtype_list=package_device_dtype_list,
    beta_candidates=beta_candidates,
    n_bootstraps=n_bootstraps,
    n_iterations=n_iterations,
)
