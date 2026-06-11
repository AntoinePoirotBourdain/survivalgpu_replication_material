import sys
import os
import numpy as np
from pathlib import Path

lib_path = os.path.abspath("/home/dev/survivalGPU/python")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from cox_functions import run_cox_experiment


# Each spec describes the covariates of one dataset as a list of (kind, HR)
# tuples, where kind is "constant" or "time_dep".
covariate_specs = [
    [("constant", 1.5)],
]


########## Figure 2

experiment_name = "figure_2"
n_patients_list = [250, 500]
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
    covariate_specs=covariate_specs,
    package_device_dtype_list=package_device_dtype_list,
    n_bootstraps=n_bootstraps,
    n_iterations=n_iterations,
)


########### Figure 3

experiment_name = "figure_3"
n_patients_list = [250, 500]
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
    covariate_specs=covariate_specs,
    package_device_dtype_list=package_device_dtype_list,
    n_bootstraps=n_bootstraps,
    n_iterations=n_iterations,
)


############ Figure 4

experiment_name = "figure_4"
n_patients_list = [250, 500]
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
    covariate_specs=covariate_specs,
    package_device_dtype_list=package_device_dtype_list,
    n_bootstraps=n_bootstraps,
    n_iterations=n_iterations,
)
