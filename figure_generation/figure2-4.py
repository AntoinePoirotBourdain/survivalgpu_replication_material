import time
import numpy as np
from pathlib import Path

from cox_functions import run_cox_experiment


hr_candidates   = [0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0]
beta_candidates = [np.log(hr) for hr in hr_candidates]
seed            = 42


######### Figure 2

experiment_name = "figure_2"
n_patients_list = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
nbootstraps = 0
output_folder = Path("results/")
n_iterations = 3
covariate_combination = [
    (0, 1),  # 1 time-dependent covariate
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
    covariate_combination=covariate_combination,
    beta_candidates=beta_candidates,
    package_device_dtype_list=package_device_dtype_list,
    nbootstraps=nbootstraps,
    n_iterations=n_iterations,
    seed=seed,
)
print(f"\nFigure 2 experiment took {time.time() - t_start:.2f} seconds")


########### Figure 3

experiment_name = "figure_3"
n_patients_list = [500, 1000, 5000, 10000,100000]
nbootstraps = 1000
batchsize = 25
output_folder = Path("results/")
n_iterations = 3
covariate_combination = [
    (0, 1),  # 1 time-dependent covariate
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
    covariate_combination=covariate_combination,
    beta_candidates=beta_candidates,
    package_device_dtype_list=package_device_dtype_list,
    nbootstraps=nbootstraps,
    batchsize=batchsize,
    n_iterations=n_iterations,
    seed=seed,
)
print(f"\nFigure 3 experiment took {time.time() - t_start:.2f} seconds")


############ Figure 4

experiment_name = "figure_4"
# n_patients_list = [500, 1000, 5000, 10000, 100000, 500000, 1000000]
n_patients_list = [500, 1000, 5000, 10000, 100000, 500000, 1000000]

nbootstraps = 0
output_folder = Path("results/")
n_iterations = 3
covariate_combination = [
    (1, 0),  # 1 constant covariate
]


package_device_dtype_list = [
    ("survivalgpu", "cuda", np.float32),
    ("survival", "cpu", np.float64),
    ("xgboost", "cuda", np.float32),
    ("torchsurv", "cuda", np.float32),
]

t_start = time.time()
run_cox_experiment(
    experiment_name=experiment_name,
    output_folder=output_folder,
    n_patients_list=n_patients_list,
    covariate_combination=covariate_combination,
    beta_candidates=beta_candidates,
    package_device_dtype_list=package_device_dtype_list,
    nbootstraps=nbootstraps,
    n_iterations=n_iterations,
    seed=seed,
)
print(f"\nFigure 4 experiment took {time.time() - t_start:.2f} seconds")
