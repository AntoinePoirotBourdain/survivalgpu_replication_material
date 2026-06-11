from itertools import product

import pandas as pd 
import numpy as np
from zmq import device
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import time
import sys
import os
from pathlib import Path
from itertools import product

lib_path = os.path.abspath("/home/dev/survivalGPU/python")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import torch as torch
from survivalgpu import WCESurvivalAnalysis, simulate_for_experiment


def wce_survivalgpu(start, stop, patient_id, event, doses, constrained, dataset, n_knots, cutoff,
                     batch_size = 1, n_bootstraps = 1, device ="cpu", dtype = np.float32, covariates = None):

    patient_id = np.array(dataset[patient_id], dtype=np.int64)
    start = np.array(dataset[start], dtype=np.int64)
    stop = np.array(dataset[stop], dtype=np.int64)
    event = np.array(dataset[event], dtype=np.int64)
    doses = np.array(dataset[doses], dtype=np.float64)
    if covariates is not None:
        covariates = np.array(dataset[covariates], dtype=np.float64)


    model = WCESurvivalAnalysis(
        cutoff=cutoff,
        n_knots=n_knots,
        dtype = dtype,
        device = device,
        constrained = constrained,
        n_bootstraps = n_bootstraps,
        batch_size = batch_size,
    )

    time_start = time.time()

    model.fit(
        start = start,
        stop = stop,
        patient = patient_id,
        event = event,
        dose = doses,
        covariates = covariates,
    )

    

    time_stop = time.time()
    computation_time = time_stop - time_start

    coef_list = model.coef_[0].tolist()

    vecnum = np.ones(cutoff)
    vecdenom = np.zeros(cutoff)

    HR = model.HR(vecnum, vecdenom, 0.95)

    BIC = model.BIC_


    return HR, coef_list, computation_time, BIC


def wce_WCE(start, stop, patient_id, event, dose, 
            constrained,dataset,n_knots, cutoff,covariates = None,
            ):
    
    
    ro.r("library(WCE)")

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["R_dataset"] = ro.conversion.py2rpy(dataset)

    
    time_start = time.time()

    ro.r(f"""
        model_wce <- WCE(
            data        = R_dataset,
            nknots      = {n_knots},
            cutoff      = {cutoff},
            id          = "{patient_id}",
            event       = "{event}",
            start       = "{start}",
            stop        = "{stop}",
            expos       = "{dose}",
            constrained = "{constrained}"
        )
    """)

    time_stop = time.time()
    computation_time = time_stop - time_start

    if covariates is not None:
        coef_list = ro.r("coef(model_wce)")[1:].tolist()  # Exclude the first coefficient which corresponds to the WCE term
    else:
        coef_list = []

  
    hr_result = ro.r(f"""
        exposed   <- rep(1, {cutoff})
        unexposed <- rep(0, {cutoff})
        WCE::HR.WCE(model_wce, exposed, unexposed)
    """)

    BIC = ro.r("model_wce$info.criterion[[1]]")[0]


    return hr_result, coef_list, computation_time, BIC


# def wce_wceGPU(start, stop, patient_id, event, dose,
#                constrained, dataset, ties, n_knots, cutoff,
#                batch_size=0, n_bootstraps=0, double_precision=True,
#                aic=False, confint=0.95, verbosity=0, covariates=None):

#     ro.r(f'devtools::load_all("/home/dev/survivalGPU/R")')

#     with localconverter(ro.default_converter + pandas2ri.converter):
#         ro.globalenv["R_dataset"] = ro.conversion.py2rpy(dataset)

#     # Convert constrained: Python string "right"/"left"/False → R value
#     if isinstance(constrained, bool):
#         r_constrained = "TRUE" if constrained else "FALSE"
#     else:
#         r_constrained = f'"{constrained}"'

#     # Convert covariates: None → NULL, single string → c("col"), list → c("col1", "col2")
#     if covariates is None:
#         r_covariates = "NULL"
#     elif isinstance(covariates, str):
#         r_covariates = f'c("{covariates}")'
#     else:
#         quoted = ", ".join(f'"{c}"' for c in covariates)
#         r_covariates = f"c({quoted})"

#     r_aic = "TRUE" if aic else "FALSE"
#     r_double = "TRUE" if double_precision else "FALSE"

#     time_start = time.time()

#     ro.r(f"""
#         model_wcegpu <- wceGPU(
#             data             = R_dataset,
#             nknots           = {n_knots},
#             cutoff           = {cutoff},
#             id               = "{patient_id}",
#             event            = "{event}",
#             start            = "{start}",
#             stop             = "{stop}",
#             expos            = "{dose}",
#             covariates       = {r_covariates},
#             constrained      = {r_constrained},
#             aic              = {r_aic},
#             confint          = {confint},
#             nbootstraps      = {n_bootstraps},
#             batchsize        = {batch_size},
#             verbosity        = {verbosity},
#             double_precision = {r_double}
#         )
#     """)

#     time_stop = time.time()
#     computation_time = time_stop - time_start

#     coef_list = list(ro.r("as.vector(model_wcegpu$beta.hat)"))

#     hr_result = ro.r(f"""
#         exposed   <- rep(1, {cutoff})
#         unexposed <- rep(0, {cutoff})
#         WCE::HR.WCE(model_wcegpu, exposed, unexposed)
#     """)

#     return hr_result, coef_list, computation_time


# TODO add the BIC
def run_package_wce(package_name,
        start, stop, patient_id, event, dose, constrained, dataset, n_knots_list, cutoff, dtype,
                     batch_size = 0, n_bootstraps = 0, device ="cpu", covariates = None ):
    
    time_total = 0 

    hr_result_list = []
    coef_list_list = []
    bic_list = []

    
    if package_name == "survivalgpu":


        for knots in n_knots_list:
            hr_result, coef_list, computation_time, BIC = wce_survivalgpu(
                start=start,
                stop=stop,
                patient_id=patient_id,
                event=event,
                doses=dose,
                constrained=constrained,
                dataset=dataset,
                n_knots=knots,
                cutoff=cutoff,
                batch_size = batch_size,
                n_bootstraps = n_bootstraps,
                device = device,
                dtype = dtype,
                covariates = covariates
            )
            time_total += computation_time

            hr_result_list.append(hr_result)
            coef_list_list.append(coef_list)
            bic_list.append(BIC)

        



    elif package_name == "WCE":

        time_total = 0 

        if constrained is None:
            constrained = "FALSE"



        for knots in n_knots_list:
            hr_result, coef_list, computation_time, BIC = wce_WCE(
                start=start,
                stop=stop,
                patient_id=patient_id,
                event=event,
                dose=dose,
                constrained=constrained,
                dataset=dataset,
                n_knots=knots,
                cutoff=cutoff,
                covariates=covariates
            )
            time_total += computation_time

            hr_result_list.append(hr_result)
            coef_list_list.append(coef_list)
            bic_list.append(BIC)



    best_bic_index = bic_list.index(min(bic_list))
    hr_result = hr_result_list[best_bic_index]
    coef_list = coef_list_list[best_bic_index]


    return hr_result, coef_list, computation_time


def run_benchmark_bootstraps(package_name,
                            start, stop, patient_id, event, dose, constrained, dataset, n_knots_list, cutoff, dtype,
                            batch_size = 0, n_bootstraps = 0, device ="cpu", covariates = None):

    if n_bootstraps > 0 and package_name == "survivalgpu" and device == "cuda":
        return run_package_wce(
            package_name, start, stop, patient_id, event, dose, constrained, dataset,
            n_knots_list, cutoff, dtype,
            batch_size=batch_size, n_bootstraps=n_bootstraps, device="cuda", covariates=covariates,
        )

    # For all other cases (survivalgpu CPU or WCE with bootstraps, or no bootstraps at all):
    # run a single iteration without bootstraps and extrapolate if needed.
    hr_result, coef_list, computation_time = run_package_wce(
        package_name, start, stop, patient_id, event, dose, constrained, dataset,
        n_knots_list, cutoff, dtype,
        batch_size=0, n_bootstraps=0, device=device, covariates=covariates,
    )

    if n_bootstraps > 0:
        computation_time = computation_time * (n_bootstraps + 1)

    return hr_result, coef_list, computation_time
    

def validation_wce_experiment(
    experiment_name,
    output_folder,
    n_iteration,
    n_patients,
    max_time,
    n_knots_list,
    constraint_list,
    cutoff_list,
    HR_list,
    scenario_list,
    dtype,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{experiment_name}.csv"

    rows = []

    for (scenario, HR_target, constraint, cutoff, n_knots) in product(scenario_list, HR_list, constraint_list, cutoff_list, n_knots_list):
        print(f"Running scenario {scenario} with target HR {HR_target}, constraint {constraint}, cutoff {cutoff} and n_knots {n_knots}")
        for iteration in range(1, n_iteration + 1):
            print(f"Iteration {iteration}/{n_iteration}...")

            dataset = simulate_for_experiment(n_patients, max_time, HR_target, scenario)

            HR_survivalgpu, coef_survivalgpu, time_survivalgpu = run_package_wce(
                package_name = "survivalgpu",
                start = "start",
                stop = "stop",
                patient_id = "patients",
                event = "events",
                dose = "dose",
                constrained = constraint,
                dataset = dataset,
                n_knots_list = [n_knots],
                cutoff = cutoff,
                batch_size = 0,
                n_bootstraps = 0,
                device = "cuda",
                dtype = dtype,
            )

            HR_WCE, coef_WCE, time_WCE = run_package_wce(
                package_name = "WCE",
                start = "start",
                stop = "stop",
                patient_id = "patients",
                event = "events",
                dose = "dose",
                constrained = constraint,
                dataset = dataset,
                n_knots_list = [n_knots],
                cutoff = cutoff,
                dtype = np.float64,
            )

            HR_diff = HR_survivalgpu["HR"] - HR_WCE[0]

            print(f"HR difference between survivalgpu and WCE: {HR_diff}")

            constraint_str = constraint if constraint is not None else "None"

            rows.append({
                "experiment_name": experiment_name,
                "scenario": scenario,
                "HR_target": HR_target,
                "iteration": iteration,
                "constraint": constraint_str,
                "cutoff": cutoff,
                "n_knots": n_knots,
                "HR_survivalgpu": HR_survivalgpu["HR"],
                "coef_survivalgpu": coef_survivalgpu,
                "time_survivalgpu": time_survivalgpu,
                "HR_WCE": HR_WCE[0],
                "coef_WCE": coef_WCE,
                "absolute_difference": abs(HR_diff),
                "relative_difference": abs(HR_diff) / abs(HR_WCE[0]),
            })

            pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)

    df_summary = (
        df.groupby(["scenario", "HR_target", "constraint", "cutoff", "n_knots"], as_index=False)
        .agg(
            median_relative_difference=("relative_difference", "median"),
            max_relative_difference=("relative_difference", "max"),
            n_simulations=("iteration", "count"),
        )
    )
    summary_path = output_folder / f"{experiment_name}_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary results saved to {summary_path}")
    print(f"Validation results saved to {output_path}")

    return df, df_summary


def validation_wce(n_iteration,n_patients, max_time, constraint_list, cutoff_list, HR_list, scenario_list, dtype):

    rows = []

    for (scenario, HR_target, constraint, cutoff) in product(scenario_list, HR_list, constraint_list, cutoff_list):
        print(f"Running scenario {scenario} with target HR {HR_target}, constraint {constraint} and cutoff {cutoff}")
        for iteration in range(1, n_iteration + 1):
            print(f"Iteration {iteration}/{n_iteration}...")

            dataset = simulate_for_experiment(n_patients, max_time,HR_target, scenario)
            HR_survivalgpu, coef_survivalgpu, time_survivalgpu = run_package_wce(
                package_name = "survivalgpu",
                start = "start",
                stop = "stop",
                patient_id = "patients",
                event = "events",
                dose = "dose",
                constrained = constraint,
                dataset = dataset,
                n_knots_list = [1,2,3],
                cutoff = cutoff,
                batch_size = 0,
                n_bootstraps = 0,
                device = "cuda",
                dtype=dtype,
                )

            HR_WCE, coef_WCE, time_WCE = run_package_wce(
                package_name = "WCE",
                start = "start",
                stop = "stop",
                patient_id = "patients",
                event = "events",
                dose = "dose",
                constrained = constraint,
                dataset = dataset,
                n_knots_list = [1,2,3],
                cutoff = cutoff,
                dtype = np.float64
                # covariates = "covariate",
            )

            HR_diff  = HR_survivalgpu["HR"] - HR_WCE[0]

            print(f"HR difference between survivalgpu and WCE: {HR_diff}")

            constraint_str = constraint if constraint is not None else "None"

            rows.append({
                "scenario": scenario,
                "HR_target": HR_target,
                "iteration": iteration,
                "constraint": constraint_str,
                "cutoff": cutoff,
                "HR_survivalgpu": HR_survivalgpu["HR"],
                "coef_survivalgpu": coef_survivalgpu,
                "time_survivalgpu": time_survivalgpu,
                "HR_WCE": HR_WCE[0],
                "coef_WCE": coef_WCE,
                "absolute_difference": abs(HR_diff),
                "relative_difference": abs(HR_diff) / abs(HR_WCE[0])
            })


    return pd.DataFrame(rows)






def run_wce_experiment(
    experiment_name,
    output_folder,
    n_patients_list,
    package_dtype_list,
    scenario,
    HR_target,
    max_time,
    cutoff,
    n_knots_list,
    constrained,
    n_bootstraps=0,
    batch_size=0,
    n_iterations=3,
    seed=None,
):
    """Run a WCE benchmark experiment.

    Parameters
    ----------
    package_dtype_list : list of (package_name, dtype, device) tuples
        E.g. [("survivalgpu", np.float32, "cuda"), ("WCE", np.float64, "cpu")]
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{experiment_name}.csv"

    df_results = []

    print(f"\nStarting the WCE experiment '{experiment_name}' with scenario='{scenario}', HR_target={HR_target}, cutoff={cutoff}, n_knots_list={n_knots_list}, constrained={constrained}, n_bootstraps={n_bootstraps}, batch_size={batch_size}")

    for dataset_idx, n_patients in enumerate(n_patients_list):
        dataset_seed = None if seed is None else seed + dataset_idx
        t_sim_start = time.time()
        dataset = simulate_for_experiment(
            n_patients=n_patients,
            max_time=max_time,
            HR_target=HR_target,
            scenario_name=scenario,
            seed=dataset_seed,
        )
        t_sim = time.time() - t_sim_start
        print(f"\nn_patients={n_patients} | scenario={scenario} | "
              f"HR_target={HR_target} | cutoff={cutoff}")
        print(f"  dataset: {len(dataset)} rows, {dataset['patients'].nunique()} patients "
              f"| simulation time: {t_sim:.4f} seconds")

        for package, dtype, device in package_dtype_list:
            print(f"  running {package} | dtype={dtype.__name__} | device={device} | n_iterations={n_iterations} ...")

            # survivalgpu+cuda runs the full n_bootstraps natively; all other cases
            # are already extrapolated inside run_benchmark_bootstraps.

            total_time = 0
            for i in range(n_iterations):
                print(f"    iteration {i+1}/{n_iterations}...")
                hr_result, coef_list, iter_time = run_benchmark_bootstraps(
                    package_name=package,
                    start="start",
                    stop="stop",
                    patient_id="patients",
                    event="events",
                    dose="dose",
                    constrained=constrained,
                    dataset=dataset,
                    n_knots_list=n_knots_list,
                    cutoff=cutoff,
                    dtype=dtype,
                    batch_size=batch_size,
                    n_bootstraps=n_bootstraps,
                    device=device,
                )
                total_time += iter_time

            computation_time = total_time / n_iterations
            print(f"  mean time over {n_iterations} iterations: {computation_time:.4f} seconds")

            df_results.append({
                "experiment_name" : experiment_name,
                "package"         : package,
                "n_patients"      : n_patients,
                "scenario"        : scenario,
                "HR_target"       : HR_target,
                "max_time"        : max_time,
                "cutoff"          : cutoff,
                "constrained"     : constrained,
                "dtype"           : dtype.__name__,
                "device"          : device,
                "computation_time": computation_time,
            })

            pd.DataFrame(df_results).to_csv(output_path, index=False)

    return pd.DataFrame(df_results)


# def run_wce_benchmark_experiment():
    




        


        
        


        
        

    

    
        


    