# import time
# import pandas as pd
# import torch
# import torch.nn as nn
# import numpy as np
# from rpy2.robjects.packages import importr
# from torchsurv.loss.cox import neg_partial_log_likelihood

import itertools
from unittest import result

import pandas as pd 
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import time
import sys
import os
from pathlib import Path

from simulation_functions import quick_simulated_dataset


## pacakges import for benchmarks
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
import torch.nn as nn
from torchsurv.loss.cox import neg_partial_log_likelihood
import xgboost as xgb






lib_path = os.path.abspath("/home/dev/survivalGPU/python")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import torch as torch
from survivalgpu import CoxPHSurvivalAnalysis, WCESurvivalAnalysis, simulate_dataset, ConstantCovariate, TimeDependentCovariate


CUDA_PACKAGES = ["survivalgpu", "torchsurv","xgboost"]

_CONSTANT_COVARIATES = [
    ConstantCovariate("constant_1",  coef=np.log(1.5),  values=[0, 1],          weights=[0.5, 0.5]),
    ConstantCovariate("constant_2",  coef=np.log(1.2),  values=[0, 1, 2],       weights=[0.3, 0.4, 0.3]),
    ConstantCovariate("constant_3",  coef=np.log(0.8),  values=[0, 1],          weights=[0.7, 0.3]),
    ConstantCovariate("constant_4",  coef=np.log(1.3),  values=[0, 1, 2, 3],    weights=[0.1, 0.2, 0.3, 0.4]),
    ConstantCovariate("constant_5",  coef=np.log(0.7),  values=[0, 1],          weights=[0.6, 0.4]),
    ConstantCovariate("constant_6",  coef=np.log(1.4),  values=[0, 1, 2],       weights=[0.2, 0.5, 0.3]),
    ConstantCovariate("constant_7",  coef=np.log(0.9),  values=[0, 1],          weights=[0.8, 0.2]),
    ConstantCovariate("constant_8",  coef=np.log(1.1),  values=[0, 1, 2, 3],    weights=[0.25, 0.25, 0.25, 0.25]),
    ConstantCovariate("constant_9",  coef=np.log(0.85), values=[0, 1],          weights=[0.55, 0.45]),
    ConstantCovariate("constant_10", coef=np.log(1.25), values=[0, 1, 2],       weights=[0.4, 0.4, 0.2]),
]

_TD_COVARIATES = [
    TimeDependentCovariate("time_dep_1",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(2.0)),
    TimeDependentCovariate("time_dep_2",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(0.5)),
    TimeDependentCovariate("time_dep_3",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(1.5)),
    TimeDependentCovariate("time_dep_4",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(0.7)),
    TimeDependentCovariate("time_dep_5",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(1.8)),
    TimeDependentCovariate("time_dep_6",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(0.6)),
    TimeDependentCovariate("time_dep_7",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(1.3)),
    TimeDependentCovariate("time_dep_8",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(0.8)),
    TimeDependentCovariate("time_dep_9",  values=[0.5, 1.0, 1.5, 2.0], coef=np.log(1.6)),
    TimeDependentCovariate("time_dep_10", values=[0.5, 1.0, 1.5, 2.0], coef=np.log(0.9)),
]


def open_dataset(n_patients, n_constant_covariates, n_time_dependent_covariates, compressed = True):

    if n_constant_covariates > 0 and n_time_dependent_covariates > 0:
        filename = f"{n_patients}_{n_constant_covariates}c_{n_time_dependent_covariates}td.csv"
    elif n_constant_covariates == 0 and n_time_dependent_covariates > 0:
        filename = f"{n_patients}_{n_time_dependent_covariates}td.csv"
    elif n_constant_covariates > 0 and n_time_dependent_covariates == 0:
        filename = f"{n_patients}_{n_constant_covariates}c.csv"
    else:
        raise ValueError("At least one of n_constant_covariates and n_time_dependent_covariates must be greater than 0")



    if compressed:
        path = os.path.join("/home/dev/analysis_survivalgpu/JSS/benchmark_datasets_compressed", filename)
    else:
        path = os.path.join("/home/dev/analysis_survivalgpu/JSS/benchmark_datasets", filename)

    return pd.read_csv(path)


def cox_r_survival(start, stop, event, covariates, dataset, ties):

    # Get covaraite formulat from covariate names
    covariate_formula = " + ".join(covariates)

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dataset = ro.conversion.py2rpy(dataset)

    ro.globalenv["dataset"] = r_dataset

    time_start = time.time()
    ro.r("library(survival)")
    if start == None:
        ro.r(f"fit <- coxph(Surv({stop}, {event}) ~ {covariate_formula}, data=dataset, ties='{ties}')")
    else:
        ro.r(f"fit <- coxph(Surv({start}, {stop}, {event}) ~ {covariate_formula}, data=dataset, ties='{ties}')")
    time_stop = time.time()
    computation_time = time_stop - time_start

    coef_list = [ro.r("coef(fit)")[i] for i in range(len(covariates))]
    return coef_list, computation_time



def cox_survivalgpu(start, stop, event, covariates, dataset, ties, batch_size = 0, n_bootstraps = 0, device ="cpu", dtype = np.float32):

    start = np.array(dataset[start], dtype=np.int64) if start is not None else np.zeros(len(dataset), dtype=np.int64)
    stop = np.array(dataset[stop], dtype=np.int64)
    event = np.array(dataset[event], dtype=np.int64)
    # covariate is a list of covariate names, we need to get the corresponding columns from the dataset and put them in a numpy array
    covariates = np.array(dataset[covariates], dtype=np.float64)


    model = CoxPHSurvivalAnalysis(
        ties=ties,
        dtype=dtype,
        device=device,
        n_bootstraps=n_bootstraps,
        batch_size=batch_size,
    )

    time_start = time.time()

    model.fit(start = start, stop = stop, event = event, covariates = covariates)


    time_stop = time.time()
    computation_time = time_stop - time_start

    coef_list = model.coef_[0].tolist()
    return coef_list, computation_time




def cox_lifelines (start, stop, event, covariates, dataset, ties):
    """Fit a Cox model with lifelines.
    """

    if start == None:
        model = CoxPHFitter(baseline_estimation_method="breslow")
        time_start = time.time()
        model.fit(
            df = dataset,
            duration_col = stop,
            event_col = event,
            formula = " + ".join(covariates)
        )
        time_stop = time.time()
        computation_time = time_stop - time_start

        coef_list = model.params_.tolist()

    else:
        if ties == "breslow":
            # liflines does not support breslow ties method for time-varying covariates, error
            error_message = "Lifelines does not support breslow ties method for time-varying covariates"
            raise ValueError(error_message)
        model = CoxTimeVaryingFitter()
        time_start = time.time()
        model.fit(
            df = dataset,
            start_col = start,
            stop_col = stop,
            event_col = event,
            formula = " + ".join(covariates)
        )
        time_stop = time.time()
        computation_time = time_stop - time_start

        coef_list = model.params_.tolist()

    return coef_list, computation_time




def cox_torchsurv(start, stop, event, covariates, dataset, ties, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X  = torch.tensor(dataset[covariates].values, dtype=torch.float32).to(device)
    t  = torch.tensor(dataset[stop].values,       dtype=torch.float32).to(device)
    ev = torch.tensor(dataset[event].values,      dtype=torch.bool).to(device)

    if start is not None:
        t0 = torch.tensor(dataset[start].values,  dtype=torch.float32).to(device)

    model = nn.Linear(X.shape[1], 1, bias=False).to(device)
    nn.init.zeros_(model.weight)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        tolerance_grad=1e-7, tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()

        if start is None:
            log_hz = model(X).squeeze(1)
            loss = neg_partial_log_likelihood(log_hz, ev, t, ties_method=ties)
        else:
            log_hz_vec = model(X).squeeze(1)
            log_hz_mat = log_hz_vec.unsqueeze(1).expand(-1, len(t)).contiguous()

            at_risk = (t0.unsqueeze(1) < t.unsqueeze(0)) & \
                    (t.unsqueeze(0)  <= t.unsqueeze(1))

            log_hz_mat = log_hz_mat.masked_fill(~at_risk, float('-inf'))
            loss = neg_partial_log_likelihood(log_hz_mat, ev, t, ties_method=ties)

        loss.backward()
        return loss

    t0_time = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - t0_time

    coef_list = model.weight.data.squeeze().tolist()
    return coef_list, elapsed


def cox_statsmodels(start, stop, event, covariates, dataset, ties):
    print( "cox_statsmodels is not implemented yet")

def cox_xgboost(start, stop, event, covariates, dataset, device="cpu"):

    if start is not None:
        raise ValueError("cox_xgboost does not support time-varying covariates yet")

    X     = dataset[covariates].values
    t     = dataset[stop].values
    ev    = dataset[event].values.astype(int)

    # XGBoost Cox: positive = event, negative = censored
    label = np.where(ev == 1, t, -t)

    dtrain = xgb.DMatrix(X, label=label)

    params = {
        "objective"        : "survival:cox",
        "eval_metric"      : "cox-nloglik",
        "booster"          : "gblinear",
        "lambda"           : 0,
        "alpha"            : 0,
        "feature_selector" : "shuffle",
        "updater"          : "coord_descent",
        "device"           : device,
    }

    t0 = time.time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round = 20,
        verbose_eval    = False,
    )
    elapsed = time.time() - t0

    coef_list = [model.get_score(importance_type="weight").get(f"f{i}", 0.0)
                 for i in range(len(covariates))]

    return coef_list, elapsed


def run_package_cox(
        package_name, 
        start, stop, event, covariates, dataset, ties,
        batch_size = 0, n_bootstraps = 0, device ="cpu", dtype = np.float32
):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine")
    if device == "cuda" and package_name not in CUDA_PACKAGES:
        raise ValueError(f"Package {package_name} does not support GPU") 
    
    if package_name == "survival":
        return cox_r_survival(start, stop, event, covariates, dataset, ties)
    
    if package_name == "survivalgpu":
        return cox_survivalgpu(start, stop, event, covariates, dataset, ties, batch_size, n_bootstraps, device, dtype)

    if package_name == "lifelines":
        return cox_lifelines(start, stop, event, covariates, dataset, ties)
    
    if package_name == "torchsurv":
        return cox_torchsurv(start, stop, event, covariates, dataset, ties="efron", device=device)
    if package_name == "xgboost":
        return cox_xgboost(start, stop, event, covariates, dataset)
    
    raise ValueError(f"Package {package_name} not recognized")



def run_benchmark_bootstraps(
        package_name,
        start, stop, event, covariates, dataset, ties,
        batch_size = 0, n_bootstraps = 0, device ="cpu", dtype = np.float32
):
    if n_bootstraps > 0 and package_name == "survivalgpu" and device == "cuda":
        return cox_survivalgpu(start, stop, event, covariates, dataset, ties, batch_size, n_bootstraps, device, dtype)

    # For all other cases (survivalgpu CPU, other packages, or no bootstraps):
    # run a single iteration without bootstraps and extrapolate if needed.
    coef_list, computation_time = run_package_cox(
        package_name=package_name,
        start=start,
        stop=stop,
        event=event,
        covariates=covariates,
        dataset=dataset,
        ties=ties,
        batch_size=0,
        n_bootstraps=0,
        device=device,
        dtype=dtype,
    )

    if n_bootstraps > 0:
        computation_time = computation_time * (n_bootstraps + 1)

    return coef_list, computation_time





# def run_benchmark(
#     n_patients_list,
#     n_constant_covariates,
#     n_time_dependant_covariates, 
#     package,
#     n_bootstraps=0,
#     batch_size=0,
#     device="cpu",
# ):
#     df_result = pd.DataFrame(columns=[
#         "package", "n_patients", "n_constant_covariates", "n_time_dependent_covariates", "device", "computation_time"
#     ])

#     for n_patients in n_patients_list:
#         dataset = open_dataset(
#             n_patients=n_patients,                                    # ← was n_patients_list[0]
#             n_constant_covariates=n_constant_covariates,
#             n_time_dependent_covariates=n_time_dependant_covariates,
#             compressed=True,
#         )

#         print(f"Running {package} on {n_patients} patients with {n_constant_covariates} constant covariates and {n_time_dependant_covariates} time-dependant covariates ...")   # ← was n_patients_list[0]

#         result = run_benchmark_bootstraps(
#             package_name=package,
#             start="start" if n_time_dependant_covariates > 0 else None,
#             stop="stop",
#             event="events",
#             covariates=[f"constant_{i+1}" for i in range(n_constant_covariates)] + [f"time_dep_{i+1}" for i in range(n_time_dependant_covariates)],
#             dataset=dataset,
#             ties="breslow",
#             batch_size=batch_size,
#             n_bootstraps=n_bootstraps,
#             device=device,
#         )
#         print(f"coef: {result[0]}, time: {result[1]:.4f} seconds")

#         df_result = pd.concat([df_result, pd.DataFrame([{      # ← .append() is removed in pandas 2.0+
#             "package": package,
#             "n_patients": n_patients,
#             "n_constant_covariates": n_constant_covariates,
#             "n_time_dependent_covariates": n_time_dependant_covariates,
#             "device": device,
#             "computation_time": result[1],
#         }])], ignore_index=True)

#     return df_result


def run_cox_experiment(
    experiment_name,
    output_folder,
    n_patients_list,
    covariate_combination,
    package_device_dtype_list,
    n_bootstraps=0,
    batch_size=0,
    n_iterations=3,
    max_time=365,
    seed=None,
):
    """Run a Cox PH benchmark experiment.

    Parameters
    ----------
    package_device_dtype_list : list of (package_name, device, dtype) tuples
        E.g. [("survivalgpu", "cuda", np.float32), ("survival", "cpu", np.float64)]
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{experiment_name}.csv"

    df_results = []

    print(f"\nStarting Cox experiment '{experiment_name}' | n_bootstraps={n_bootstraps} | n_iterations={n_iterations}")

    for dataset_idx, (n_patients, (n_constant, n_time_dep)) in enumerate(
        itertools.product(n_patients_list, covariate_combination)
    ):
        covariate_list = (
            _CONSTANT_COVARIATES[:n_constant]
            + _TD_COVARIATES[:n_time_dep]
        )
        covariate_names = (
            [f"constant_{j+1}" for j in range(n_constant)]
            + [f"time_dep_{j+1}" for j in range(n_time_dep)]
        )
        dataset_seed = None if seed is None else seed + dataset_idx

        t_sim_start = time.time()
        dataset = simulate_dataset(
            max_time=max_time,
            n_patients=n_patients,
            list_covariates=covariate_list,
            compress=True,
            seed=dataset_seed,
        )
        t_sim = time.time() - t_sim_start

        print(f"\nn_patients={n_patients} | n_constant={n_constant} | n_time_dep={n_time_dep}")
        print(f"  dataset: {len(dataset)} rows | simulation time: {t_sim:.4f} seconds")

        for package, device, dtype in package_device_dtype_list:
            print(f"  running {package} | dtype={dtype.__name__} | device={device} | n_iterations={n_iterations} ...")

            total_time = 0
            for i in range(n_iterations):
                print(f"    iteration {i+1}/{n_iterations}...")
                coef_list, iter_time = run_benchmark_bootstraps(
                    package_name = package,
                    start        = "start" if n_time_dep > 0 else None,
                    stop         = "stop",
                    event        = "events",
                    covariates   = covariate_names,
                    dataset      = dataset,
                    ties         = "breslow",
                    batch_size   = batch_size,
                    n_bootstraps = n_bootstraps,
                    device       = device,
                    dtype        = dtype,
                )
                total_time += iter_time

            computation_time = total_time / n_iterations
            print(f"  mean time over {n_iterations} iterations: {computation_time:.4f} seconds")

            df_results.append({
                "experiment_name"             : experiment_name,
                "package"                     : package,
                "n_patients"                  : n_patients,
                "n_constant_covariates"       : n_constant,
                "n_time_dependent_covariates" : n_time_dep,
                "device"                      : device,
                "dtype"                       : dtype.__name__,
                "computation_time"            : computation_time,
            })

            pd.DataFrame(df_results).to_csv(output_path, index=False)

    return pd.DataFrame(df_results)




# def validation_experiment(
#         experiment_name,
#         package_list,
#         ties,
#         reference_package,
#         n_simulations, 
#         n_patients, 
#         list_covariate_combination,
#         beta_list,
# ): 
#     df_results = []

#     for n_constant, n_time_dep in list_covariate_combination:
#         print(f"Validation experiment | reference={reference_package} | ties={ties} | n_constant={n_constant} | n_time_dep={n_time_dep}")

#         covariates = [f"constant_{i+1}" for i in range(n_constant)] + [f"time_dep_{i+1}" for i in range(n_time_dep)]

#         for sim in range(n_simulations):
#             print(f"Simulation {sim+1}/{n_simulations} ...")

#             dataset = quick_simulated_dataset(
#                 n_patients=n_patients,
#                 n_constant_covariates=n_constant,
#                 n_time_dependent_covariates=n_time_dep,
#                 max_time=365,
#                 beta_list=beta_list,
#             )
#             reference_result = run_package_cox(
#                 package_name=reference_package,
#                 start="start" if n_time_dep > 0 else None,
#                 stop="stop",
#                 event="events",
#                 covariates=covariates,
#                 dataset=dataset,
#                 ties=ties,
#             )

#             for package in package_list:
#                 if package == reference_package:
#                     for i, cov in enumerate(covariates):
#                         df_results.append({
#                             "simulation": sim,
#                             "package": package,
#                             "n_constant": n_constant,
#                             "n_time_dep": n_time_dep,
#                             "covariate": cov,
#                             "mae": None,
#                             "mre": None,
#                             "error": None,
#                         })
#                     continue

#                 try:
#                     result = run_package_cox(
#                         package_name=package,
#                         start="start" if n_time_dep > 0 else None,
#                         stop="stop",
#                         event="events",
#                         covariates=covariates,
#                         dataset=dataset,
#                         ties=ties,
#                     )

#                     for i, cov in enumerate(covariates):
#                         mae = abs(result[0][i] - reference_result[0][i])
#                         mre = abs(result[0][i] - reference_result[0][i]) / abs(reference_result[0][i]) if reference_result[0][i] != 0 else float('inf')

#                         df_results.append({
#                             "simulation": sim,
#                             "package": package,
#                             "n_constant": n_constant,
#                             "n_time_dep": n_time_dep,
#                             "covariate": cov,
#                             "mae": mae,
#                             "mre": mre,
#                             "error": None,
#                         })

#                 except Exception as e:
#                     print(f"Error for package {package} on sim {sim}: {e}")
#                     for cov in covariates:
#                         df_results.append({
#                             "simulation": sim,
#                             "package": package,
#                             "n_constant": n_constant,
#                             "n_time_dep": n_time_dep,
#                             "covariate": cov,
#                             "mae": None,
#                             "mre": None,
#                             "error": str(e),
#                         })

#     df_results = pd.DataFrame(df_results)
#     os.makedirs("benchmark_results", exist_ok=True)
#     output_path = f"benchmark_results/{experiment_name}.csv"
#     df_results.to_csv(output_path, index=False)
#     print(f"Validation results saved to {output_path}")


def build_covariates_with_betas(
    n_constant,
    n_time_dep,
    first_covariate_beta_list,
    other_covariates_beta_list,
):
    total_cov = n_constant + n_time_dep
    if total_cov < 1:
        raise ValueError("At least one covariate is required.")

    if n_constant > len(_CONSTANT_COVARIATES):
        raise ValueError(f"n_constant={n_constant} exceeds available templates ({len(_CONSTANT_COVARIATES)}).")
    if n_time_dep > len(_TD_COVARIATES):
        raise ValueError(f"n_time_dep={n_time_dep} exceeds available templates ({len(_TD_COVARIATES)}).")

    if len(first_covariate_beta_list) < 1:
        raise ValueError("first_covariate_beta_list must contain at least 1 value.")

    expected_other_betas = total_cov - 1
    if len(other_covariates_beta_list) < expected_other_betas:
        raise ValueError(
            f"other_covariates_beta_list must contain at least {expected_other_betas} values "
            f"(got {len(other_covariates_beta_list)})."
        )

    beta_all = [first_covariate_beta_list[0]] + list(other_covariates_beta_list[:expected_other_betas])

    covariates = []
    covariate_names = []
    true_betas = []

    k = 0
    for i in range(n_constant):
        tmpl = _CONSTANT_COVARIATES[i]
        covariates.append(
            ConstantCovariate(
                tmpl.name,
                coef=beta_all[k],
                values=tmpl.values,
                weights=tmpl.weights,
            )
        )
        covariate_names.append(tmpl.name)
        true_betas.append(beta_all[k])
        k += 1

    for i in range(n_time_dep):
        tmpl = _TD_COVARIATES[i]
        covariates.append(
            TimeDependentCovariate(
                tmpl.name,
                values=tmpl.values,
                coef=beta_all[k],
            )
        )
        covariate_names.append(tmpl.name)
        true_betas.append(beta_all[k])
        k += 1

    return covariates, covariate_names, true_betas


# ...existing code...

# ...existing code...

def validation_experiment(
    experiment_name,
    output_folder,
    n_patients_list,
    covariate_combination,
    n_simulations=1,
    ties_list=("breslow", "efron"),
    max_time=365,
    seed=None,
    dtype=np.float32,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{experiment_name}.csv"

    rows = []

    if isinstance(ties_list, str):
        ties_list = [ties_list]
    ties_list = list(ties_list)

    rng = np.random.default_rng(seed=seed)

    combo_iter = itertools.product(n_patients_list, covariate_combination)
    for combo_idx, (n_patients, (n_constant, n_time_dep)) in enumerate(combo_iter):
        total_cov = n_constant + n_time_dep
        print(f"\nn_patients={n_patients} | n_constant={n_constant} | n_time_dep={n_time_dep}")

        for sim in range(n_simulations):
            _BETA_CANDIDATES = [np.log(0.25), np.log(0.5), np.log(0.75), np.log(1.5), np.log(2.0), np.log(3.0), np.log(4.0)]
            random_betas = rng.choice(_BETA_CANDIDATES, size=total_cov, replace=True).tolist()

            cov_defs, cov_names, true_betas = build_covariates_with_betas(
                n_constant=n_constant,
                n_time_dep=n_time_dep,
                first_covariate_beta_list=[random_betas[0]],
                other_covariates_beta_list=random_betas[1:],
            )

            dataset_seed = int(rng.integers(0, 2**31))

            dataset = simulate_dataset(
                max_time=max_time,
                n_patients=n_patients,
                list_covariates=cov_defs,
                compress=True,
                seed=dataset_seed,
            )

            start_col = "start" if n_time_dep > 0 else None

            for ties in ties_list:
                beta_r, _ = run_package_cox(
                    package_name="survival",
                    start=start_col,
                    stop="stop",
                    event="events",
                    covariates=cov_names,
                    dataset=dataset,
                    ties=ties,
                )

                beta_gpu, _ = run_package_cox(
                    package_name="survivalgpu",
                    start=start_col,
                    stop="stop",
                    event="events",
                    covariates=cov_names,
                    dataset=dataset,
                    ties=ties,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=dtype,
                )

                abs_diffs = [abs(beta_gpu[i] - beta_r[i]) for i in range(len(cov_names))]
                rel_diffs = [
                    abs(beta_gpu[i] - beta_r[i]) / abs(beta_r[i]) if beta_r[i] != 0 else np.nan
                    for i in range(len(cov_names))
                ]
                mean_abs_diff = float(np.mean(abs_diffs)) if abs_diffs else np.nan
                mean_rel_diff = float(np.nanmean(rel_diffs)) if rel_diffs else np.nan

                for i, cov in enumerate(cov_names):
                    rows.append({
                        "experiment_name": experiment_name,
                        "simulation": sim,
                        "ties": ties,
                        "n_patients": n_patients,
                        "n_constant_covariates": n_constant,
                        "n_time_dependent_covariates": n_time_dep,
                        "covariate": cov,
                        "true_beta": true_betas[i],
                        "beta_survival": beta_r[i],
                        "beta_survivalgpu": beta_gpu[i],
                        "abs_diff": abs_diffs[i],
                        "rel_diff": rel_diffs[i],
                        "mean_abs_diff": mean_abs_diff,
                        "mean_rel_diff": mean_rel_diff,
                        "dtype": dtype.__name__,
                    })


                pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)

    # Summary: aggregate per simulation first, then across simulations
    df_summary = (
        df.drop_duplicates(
            subset=["simulation", "ties", "n_patients", "n_constant_covariates", "n_time_dependent_covariates"],
            keep="first"
        )
        .groupby(["ties", "n_patients", "n_constant_covariates", "n_time_dependent_covariates"], as_index=False)
        .agg(
            mean_of_mean_abs_diff=("mean_abs_diff", "mean"),
            median_of_mean_abs_diff=("mean_abs_diff", "median"),
            max_of_mean_abs_diff=("mean_abs_diff", "max"),
            mean_of_mean_rel_diff=("mean_rel_diff", "mean"),
            median_of_mean_rel_diff=("mean_rel_diff", "median"),
            max_of_mean_rel_diff=("mean_rel_diff", "max"),
            n_simulations=("simulation", "count"),
        )
    )
    summary_path = output_folder / f"{experiment_name}_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary results saved to {summary_path}")
    print(f"Validation results saved to {output_path}")

    return df, df_summary

validation_experiment(
    experiment_name="val_breslow_efron",
    output_folder="benchmark_results",
    n_patients_list=[1000],
    covariate_combination=[(1, 0), (3, 0), (0, 1), (0, 3)],
    n_simulations=100,
    ties_list=["breslow", "efron"],
    seed=123,
)
