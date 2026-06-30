from itertools import product

import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import time
from pathlib import Path

from survivalgpu import WCESurvivalAnalysis, simulate_for_experiment


def wce_survivalgpu(start, stop, patient_id, event, doses, constrained, dataset, n_knots, cutoff,
                     batch_size = 1, n_bootstraps = 1, device ="cpu", dtype = np.float32, covariates = None):
    """Fits a Weighted Cumulative Exposure (WCE) model with `survivalgpu.WCESurvivalAnalysis`.

    Args:
        start (str): name of the "start" time column.
        stop (str): name of the "stop"/event-time column.
        patient_id (str): name of the patient identifier column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        doses (str): name of the exposure/dose column.
        constrained ("right", "left" or None): shape constraint on the B-spline
            risk function, see `WCESurvivalAnalysis`.
        dataset (pandas.DataFrame): the input dataset.
        n_knots (int): number of knots for the B-splines.
        cutoff (int): size of the time window for the risk function.
        batch_size (int, optional): batch size for the fit. Defaults to 1.
        n_bootstraps (int, optional): number of bootstrap resamples. Defaults
            to 1.
        device (str, optional): "cpu" or "cuda". Defaults to "cpu".
        dtype (numpy.dtype, optional): floating point precision. Defaults to
            np.float32.
        covariates (list[str] or None, optional): names of additional
            (non-WCE) covariate columns. Defaults to None.

    Returns:
        tuple[numpy.ndarray, list[float], float, float]: a tuple of
            (HR, coef_list, computation_time, BIC), where `HR` is the hazard
            ratio curve returned by `model.HR` for an exposed-vs-unexposed
            dose of 1 over the full cutoff window, `coef_list` are the
            coefficients of the first bootstrap sample, `computation_time` is
            the fit duration in seconds, and `BIC` is the model's BIC.
    """
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
    """Fits a Weighted Cumulative Exposure (WCE) model with R's `WCE` package, via rpy2.

    Args:
        start (str): name of the "start" time column.
        stop (str): name of the "stop"/event-time column.
        patient_id (str): name of the patient identifier column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        dose (str): name of the exposure/dose column.
        constrained (str): shape constraint on the WCE risk function, passed
            as-is to `WCE::WCE` (e.g. "right", "left" or "FALSE").
        dataset (pandas.DataFrame): the input dataset.
        n_knots (int): number of knots for the WCE model.
        cutoff (int): size of the time window for the risk function.
        covariates (list[str] or None, optional): if not None, the
            coefficients of the additional covariates are extracted (excluding
            the WCE term). Defaults to None.

    Returns:
        tuple[float, list[float], float, float]: a tuple of
            (hr_result, coef_list, computation_time, BIC), where `hr_result`
            is the hazard ratio between a constant dose of 1 and 0 over the
            full cutoff window, `coef_list` are the coefficients of the
            additional covariates (empty if `covariates` is None),
            `computation_time` is the fit duration in seconds, and `BIC` is
            the model's first information criterion.
    """
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


def run_package_wce(package_name,
        start, stop, patient_id, event, dose, constrained, dataset, n_knots_list, cutoff, dtype,
                     batch_size = 0, n_bootstraps = 0, device ="cpu", covariates = None ):
    """Fits a WCE model with the requested package, selecting the best number of knots by BIC.

    For each value of `n_knots` in `n_knots_list`, fits a model with
    `wce_survivalgpu` or `wce_WCE` and records its BIC. Returns the result
    for the `n_knots` value with the lowest BIC.

    Args:
        package_name (str): either "survivalgpu" or "WCE".
        start (str): name of the "start" time column.
        stop (str): name of the "stop"/event-time column.
        patient_id (str): name of the patient identifier column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        dose (str): name of the exposure/dose column.
        constrained ("right", "left", None or str): shape constraint on the
            WCE risk function (see `wce_survivalgpu` / `wce_WCE`).
        dataset (pandas.DataFrame): the input dataset.
        n_knots_list (list[int]): candidate numbers of knots to try.
        cutoff (int): size of the time window for the risk function.
        dtype (numpy.dtype): floating point precision, only used by
            "survivalgpu".
        batch_size (int, optional): batch size, only used by "survivalgpu".
            Defaults to 0.
        n_bootstraps (int, optional): number of bootstrap resamples, only used
            by "survivalgpu". Defaults to 0.
        device (str, optional): "cpu" or "cuda", only used by "survivalgpu".
            Defaults to "cpu".
        covariates (list[str] or None, optional): names of additional
            covariate columns. Defaults to None.

    Returns:
        tuple[float or numpy.ndarray, list[float], float]: the
            (hr_result, coef_list, computation_time) of the fit with the
            lowest BIC among `n_knots_list`. Note `computation_time` is that
            of the last fit performed, not necessarily the best one.
    """
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
    """Runs a WCE fit via `run_package_wce`, optionally with bootstrapping, for benchmarking purposes.

    If `n_bootstraps > 0` and the package is "survivalgpu" running on "cuda",
    the bootstrap resamples are fit directly (a single timed call covering all
    bootstraps). Otherwise, a single fit without bootstraps is timed and, if
    `n_bootstraps > 0`, its computation time is extrapolated by multiplying by
    `n_bootstraps + 1`.

    Args:
        package_name (str): either "survivalgpu" or "WCE".
        start (str): name of the "start" time column.
        stop (str): name of the "stop"/event-time column.
        patient_id (str): name of the patient identifier column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        dose (str): name of the exposure/dose column.
        constrained ("right", "left", None or str): shape constraint on the
            WCE risk function.
        dataset (pandas.DataFrame): the input dataset.
        n_knots_list (list[int]): candidate numbers of knots, see `run_package_wce`.
        cutoff (int): size of the time window for the risk function.
        dtype (numpy.dtype): floating point precision, only used by
            "survivalgpu".
        batch_size (int, optional): batch size, only used by "survivalgpu".
            Defaults to 0.
        n_bootstraps (int, optional): number of bootstrap resamples. Defaults
            to 0.
        device (str, optional): "cpu" or "cuda". Defaults to "cpu".
        covariates (list[str] or None, optional): names of additional
            covariate columns. Defaults to None.

    Returns:
        tuple[float or numpy.ndarray, list[float], float]: the
            (hr_result, coef_list, computation_time) of the selected fit, with
            `computation_time` possibly extrapolated.
    """
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
    hr_candidates,
    scenario_list,
    dtype,
    seed=None,
):
    """Compares `survivalgpu` and R's `WCE` hazard ratios on simulated datasets.

    For each combination of `constraint_list` x `cutoff_list` x `n_knots_list`,
    runs `n_iteration` simulations with a randomly drawn scenario (from
    `scenario_list`) and target hazard ratio (from `hr_candidates`), fits a WCE
    model with both "survivalgpu" (cuda, single `n_knots` value) and "WCE" (R,
    same `n_knots` value), and records the absolute and relative differences
    between the two packages' hazard ratios at dose 1 vs 0. Results are written
    incrementally to `<output_folder>/<experiment_name>.csv`. A per-combination
    summary of the relative differences is written to
    `<output_folder>/<experiment_name>_summary.csv`.

    Args:
        experiment_name (str): name used for the output file names.
        output_folder (str or Path): directory where results are written
            (created if needed).
        n_iteration (int): number of simulations per (constraint, cutoff,
            n_knots) combination.
        n_patients (int): number of patients to simulate per iteration.
        max_time (int): maximum follow-up time for the simulated datasets.
        n_knots_list (list[int]): numbers of knots to evaluate.
        constraint_list (list): shape constraints to evaluate (e.g. "right",
            "left", None).
        cutoff_list (list[int]): cutoff (risk window) sizes to evaluate.
        hr_candidates (list[float]): candidate target hazard ratios, one is
            drawn at random for each iteration.
        scenario_list (list[str]): candidate exposure-effect scenario names,
            one is drawn at random for each iteration (see
            `simulate_for_experiment`).
        dtype (numpy.dtype): floating point precision used by "survivalgpu".
        seed (int, optional): random seed for the scenario/HR draws. Defaults
            to None (note: dataset simulation itself is not seeded).

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: the detailed per-iteration
            results, and the per-combination summary (median and max relative
            difference across iterations).
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{experiment_name}.csv"

    rows = []

    rng = np.random.default_rng(seed=seed)

    for (constraint, cutoff, n_knots) in product(constraint_list, cutoff_list, n_knots_list):
        print(f"\n{'=' * 60}")
        print(f"Running constraint={constraint}, cutoff={cutoff}, n_knots={n_knots}")
        print(f"{'=' * 60}")
        for iteration in range(1, n_iteration + 1):
            scenario = str(rng.choice(scenario_list))
            HR_target = float(rng.choice(hr_candidates))
            print(f"\n--- Iteration {iteration}/{n_iteration} | scenario={scenario} | target HR={HR_target} ---")

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

            print(f"  HR difference between survivalgpu and WCE: {HR_diff}")

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
        df.groupby(["constraint", "cutoff", "n_knots"], as_index=False)
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
    """Runs a WCE timing benchmark experiment across patient counts and packages.

    Warms up each package/dtype/device combination on a 500-patient dataset,
    then for each `n_patients` in `n_patients_list`, simulates a dataset (see
    `simulate_for_experiment`) and times `n_iterations` fits per package via
    `run_benchmark_bootstraps`, averaging the computation time. Results are
    written incrementally to `<output_folder>/<experiment_name>.csv`.

    Args:
        experiment_name (str): name used for the output file name.
        output_folder (str or Path): directory where results are written
            (created if needed).
        n_patients_list (list[int]): numbers of patients to simulate and benchmark.
        package_dtype_list (list[tuple[str, numpy.dtype, str]]): list of
            (package_name, dtype, device) tuples, e.g.
            [("survivalgpu", np.float32, "cuda"), ("WCE", np.float64, "cpu")].
        scenario (str): exposure-effect scenario name (see `simulate_for_experiment`).
        HR_target (float): target hazard ratio for the simulated "dose" covariate.
        max_time (int): maximum follow-up time for the simulated datasets.
        cutoff (int): size of the time window for the risk function.
        n_knots_list (list[int]): candidate numbers of knots, see `run_package_wce`.
        constrained ("right", "left" or None): shape constraint on the WCE
            risk function.
        n_bootstraps (int, optional): number of bootstrap resamples. Defaults to 0.
        batch_size (int, optional): batch size, only used by "survivalgpu".
            Defaults to 0.
        n_iterations (int, optional): number of timed fits per package per
            `n_patients`, averaged. Defaults to 3.
        seed (int, optional): random seed for dataset simulation. Defaults to None.

    Returns:
        pandas.DataFrame: one row per (n_patients, package) with the
            experiment parameters and the mean computation time over
            `n_iterations`.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{experiment_name}.csv"

    df_results = []

    print(f"\n{'='*60}")
    print(f"Starting the WCE experiment '{experiment_name}' with scenario='{scenario}', HR_target={HR_target}, cutoff={cutoff}, n_knots_list={n_knots_list}, constrained={constrained}, n_bootstraps={n_bootstraps}, batch_size={batch_size}")
    print(f"{'='*60}")

    print("\nWarming up packages with 500 patients...")
    warmup_dataset = simulate_for_experiment(
        n_patients=500,
        max_time=max_time,
        HR_target=HR_target,
        scenario_name=scenario,
        seed=seed,
    )
    for package, dtype, device in package_dtype_list:
        print(f"  warming up {package} | dtype={dtype.__name__} | device={device}")
        run_benchmark_bootstraps(
            package_name=package,
            start="start",
            stop="stop",
            patient_id="patients",
            event="events",
            dose="dose",
            constrained=constrained,
            dataset=warmup_dataset,
            n_knots_list=n_knots_list,
            cutoff=cutoff,
            dtype=dtype,
            batch_size=0,
            n_bootstraps=0,
            device=device,
        )

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
        print(f"\n--- n_patients={n_patients} | scenario={scenario} | "
              f"HR_target={HR_target} | cutoff={cutoff} ---")
        print(f"  dataset: {len(dataset)} rows, {dataset['patients'].nunique()} patients "
              f"| simulation time: {t_sim:.4f} seconds")

        for package, dtype, device in package_dtype_list:
            print(f"  > {package} | dtype={dtype.__name__} | device={device} | n_iterations={n_iterations}")

            # survivalgpu+cuda runs the full n_bootstraps natively; all other cases
            # are already extrapolated inside run_benchmark_bootstraps.

            total_time = 0
            for i in range(n_iterations):
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
                print(f"      iteration {i+1}/{n_iterations}: {iter_time:.4f} seconds")

            computation_time = total_time / n_iterations
            print(f"    mean time over {n_iterations} iterations: {computation_time:.4f} seconds")

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

    