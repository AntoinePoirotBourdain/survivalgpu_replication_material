import time
import numpy as np
from pathlib import Path

from cox_functions import run_cox_experiment


# Each spec describes the covariates of one dataset as a list of (kind, HR)
# tuples, where kind is "constant" or "time_dep".


########## Figure 2

experiment_name = "figure_2"
n_patients_list = [500, 1000, 5000, 10000, 100000, 500000, 1000000]
nbootstraps = 0
output_folder = Path("results/")
n_iterations = 3
covariate_specs = [
    [("time_dep", 1.5)],
]


package_device_dtype_list = [
    ("survivalgpu", "cuda", np.float32),
    ("survivalgpu", "cpu", np.float32),
    ("survival", "cpu", np.float64),
    ("lifelines", "cpu", np.float64),
]

t_start = time.time()
run_cox_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    covariate_specs=covariate_specs,
    package_device_dtype_list=package_device_dtype_list,
    nbootstraps=nbootstraps,
    n_iterations=n_iterations,
)
print(f"\nFigure 2 experiment took {time.time() - t_start:.2f} seconds")


########### Figure 3

experiment_name = "figure_3"
n_patients_list = [500, 1000, 5000, 10000,100000]
nbootstraps = 1000
batchsize = 100
output_folder = Path("results/")
n_iterations = 3
covariate_specs = [
    [("time_dep", 1.5)],
]


package_device_dtype_list = [
    ("survivalgpu", "cuda", np.float32),
    ("survivalgpu", "cpu", np.float32),
    ("survival", "cpu", np.float64),
    ("lifelines", "cpu", np.float64),
]

t_start = time.time()
run_cox_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    covariate_specs=covariate_specs,
    package_device_dtype_list=package_device_dtype_list,
    nbootstraps=nbootstraps,
    batchsize=batchsize,
    n_iterations=n_iterations,
)
print(f"\nFigure 3 experiment took {time.time() - t_start:.2f} seconds")


############ Figure 4

experiment_name = "figure_4"
n_patients_list = [500, 1000, 5000, 10000, 100000, 500000, 1000000]
nbootstraps = 0
output_folder = Path("results/")
n_iterations = 3
covariate_specs = [
    [("constant", 1.5)],
]


package_device_dtype_list = [
    ("survivalgpu", "cuda", np.float32),
    ("survival", "cpu", np.float64),
    ("xgboost", "cpu", np.float32),
    ("torchsurv", "cpu", np.float32),
]

t_start = time.time()
run_cox_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    covariate_specs=covariate_specs,
    package_device_dtype_list=package_device_dtype_list,
    nbootstraps=nbootstraps,
    n_iterations=n_iterations,
)
print(f"\nFigure 4 experiment took {time.time() - t_start:.2f} seconds")
