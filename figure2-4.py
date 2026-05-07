from cox_functions import run_experiment
import pandas as pd
import numpy as np
from pathlib import Path



########## Figure 2


experiment_name = "figure_2"
#_patients_list = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
n_patients_list = [250,500]
n_constant_covariates = 1
n_time_dependent_covariates = 0
package_list = ["survivalgpu", "torchsurv", "xgboost"] # list of packages to benchmark
device = "cuda"
n_bootstraps = 0
output_folder = Path("results/")

package_device_list = [
    ("survivalgpu", "cuda"),
    ("survivalgpu", "cpu"),
    ("survival", "cpu"),
    ("lifelines", "cpu"),
]


experiment_result_2 = run_experiment(
        experiment_name = experiment_name,
        output_folder = output_folder,
        n_patients_list = n_patients_list,
        covariate_combination = [(n_constant_covariates, n_time_dependent_covariates)],
        package_list = package_list,
        n_bootstraps=n_bootstraps
    )



########### Figure 3


experiment_name = "figure_3"
#_patients_list = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
n_patients_list = [250,500]
n_constant_covariates = 1
n_time_dependent_covariates = 0
n_bootstraps = 1000
output_folder = Path("/home/dev/analysis_survivalgpu/JSS/benchmarks/benchmark_results")

package_device_list = [
    ("survivalgpu", "cuda"),
    ("survivalgpu", "cpu"),
    ("survival", "cpu"),
    ("lifelines", "cpu"),
]



experiment_result_2 = run_experiment(
        experiment_name = experiment_name,
        output_folder = output_folder,
        n_patients_list = n_patients_list,
        covariate_combination = [(n_constant_covariates, n_time_dependent_covariates)],
        package_device_list = package_device_list,
        n_bootstraps=n_bootstraps
    )




############ Figure 4


experiment_name = "figure_4"
#_patients_list = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
n_patients_list = [250,500]
n_constant_covariates = 1
n_time_dependent_covariates = 0
package_list = ["survivalgpu", "torchsurv", "xgboost"] # list of packages to benchmark
n_bootstraps = 0
output_folder = Path("/home/dev/analysis_survivalgpu/JSS/benchmarks/benchmark_results")

package_device_list = [
    ("survivalgpu", "cuda"),
    ("survival", "cpu"),
    ("xgboost", "cpu"),
    ("torchsurv", "cpu"),
]


experiment_result_2 = run_experiment(
        experiment_name = experiment_name,
        output_folder = output_folder,
        n_patients_list = n_patients_list,
        covariate_combination = [(n_constant_covariates, n_time_dependent_covariates)],
        package_device_list = package_device_list,
        n_bootstraps=n_bootstraps
    )
