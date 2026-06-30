# import time
# import pandas as pd
# import torch
# import torch.nn as nn
# import numpy as np
# from rpy2.robjects.packages import importr
# from torchsurv.loss.cox import neg_partial_log_likelihood

import itertools

import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import time
import os
from pathlib import Path

from simulation_functions import quick_simulated_dataset


## pacakges import for benchmarks
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
import torch.nn as nn
from torchsurv.loss.cox import neg_partial_log_likelihood
import xgboost as xgb

import torch as torch
from survivalgpu import CoxPHSurvivalAnalysis, WCESurvivalAnalysis, simulate_dataset, ConstantCovariate, TimeDependentCovariate


CUDA_PACKAGES = ["survivalgpu", "torchsurv","xgboost"]

def make_constant_covariates(n, coef_list):
    """Builds a list of binary constant covariates for use with `simulate_dataset`.

    Args:
        n (int): the number of covariates to create.
        coef_list (list[float]): the Cox coefficient (log hazard ratio) for each
            covariate. Must contain at least `n` values.

    Returns:
        list[ConstantCovariate]: covariates named "constant_1", ..., "constant_n",
            each taking values 0 or 1 with equal probability (0.5/0.5).
    """
    return [
        ConstantCovariate(f"constant_{i+1}", coef=coef_list[i], values=[0, 1], weights=[0.5, 0.5])
        for i in range(n)
    ]


def make_td_covariates(n, coef_list):
    """Builds a list of time-dependent covariates for use with `simulate_dataset`.

    Args:
        n (int): the number of covariates to create.
        coef_list (list[float]): the Cox coefficient (log hazard ratio) for each
            covariate. Must contain at least `n` values.

    Returns:
        list[TimeDependentCovariate]: covariates named "time_dep_1", ..., "time_dep_n",
            each taking values among [0.5, 1.0, 1.5, 2.0].
    """
    return [
        TimeDependentCovariate(f"time_dep_{i+1}", values=[0.5, 1.0, 1.5, 2.0], coef=coef_list[i])
        for i in range(n)
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
        path = os.path.join("../analysis_survivalgpu/JSS/benchmark_datasets_compressed", filename)
    else:
        path = os.path.join("../analysis_survivalgpu/JSS/benchmark_datasets", filename)

    return pd.read_csv(path)


def cox_r_survival(start, stop, event, covariates, dataset, ties):
    """Fits a Cox PH model with R's `survival::coxph`, via rpy2.

    Args:
        start (str or None): name of the "start" time column, or None for a
            single-record-per-subject (non time-varying) dataset.
        stop (str): name of the "stop"/event-time column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        covariates (list[str]): names of the covariate columns to include in the
            model formula.
        dataset (pandas.DataFrame): the input dataset.
        ties (str): the ties handling method, e.g. "breslow" or "efron".

    Returns:
        tuple[list[float], float]: the fitted coefficients (in the same order as
            `covariates`) and the computation time in seconds.
    """
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
    """Fits a Cox PH model with `survivalgpu.CoxPHSurvivalAnalysis`.

    Args:
        start (str or None): name of the "start" time column, or None for a
            single-record-per-subject (non time-varying) dataset (an array of
            zeros is used as start times in that case).
        stop (str): name of the "stop"/event-time column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        covariates (list[str]): names of the covariate columns.
        dataset (pandas.DataFrame): the input dataset.
        ties (str): the ties handling method, e.g. "breslow" or "efron".
        batch_size (int, optional): batch size for the fit. Defaults to 0
            (no batching).
        n_bootstraps (int, optional): number of bootstrap resamples. Defaults
            to 0 (no bootstrapping).
        device (str, optional): "cpu" or "cuda". Defaults to "cpu".
        dtype (numpy.dtype, optional): floating point precision used for the
            covariates array. Defaults to np.float32.

    Returns:
        tuple[list[float], float]: the fitted coefficients of the first
            bootstrap sample (in the same order as `covariates`) and the
            computation time in seconds.
    """
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
    """Fits a Cox PH model with `lifelines`.

    Uses `CoxPHFitter` for single-record-per-subject data, or
    `CoxTimeVaryingFitter` for time-varying (start/stop) data.

    Args:
        start (str or None): name of the "start" time column, or None to fit
            with `CoxPHFitter` (single-record-per-subject data).
        stop (str): name of the "stop"/event-time/duration column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        covariates (list[str]): names of the covariate columns to include in the
            model formula.
        dataset (pandas.DataFrame): the input dataset.
        ties (str): the ties handling method, e.g. "breslow" or "efron". Only
            used when `start` is None; "breslow" is not supported by
            `CoxTimeVaryingFitter`.

    Returns:
        tuple[list[float], float]: the fitted coefficients (in the same order as
            `covariates`) and the computation time in seconds.

    Raises:
        ValueError: if `start` is not None and `ties` is "breslow", since
            `CoxTimeVaryingFitter` does not support the Breslow ties method.
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
    """Fits a Cox PH model with `torchsurv`, using a bias-free linear layer.

    The model is optimized by maximizing the Cox partial log-likelihood with
    L-BFGS. For time-varying (start/stop) data, an at-risk mask is built and
    the log hazard is broadcast across all subjects/times before computing the
    loss.

    Args:
        start (str or None): name of the "start" time column, or None for
            single-record-per-subject data.
        stop (str): name of the "stop"/event-time column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        covariates (list[str]): names of the covariate columns.
        dataset (pandas.DataFrame): the input dataset.
        ties (str): the ties handling method passed to
            `neg_partial_log_likelihood` (e.g. "efron").
        device (torch.device, optional): device to run on. Defaults to "cuda"
            if available, otherwise "cpu".

    Returns:
        tuple[list[float], float]: the fitted coefficients (in the same order as
            `covariates`) and the computation time in seconds.
    """
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
    """Placeholder for a Cox PH fit with `statsmodels`. Not implemented yet."""
    print( "cox_statsmodels is not implemented yet")

def cox_xgboost(start, stop, event, covariates, dataset, device="cpu"):
    """Fits a Cox PH model with `xgboost`'s `survival:cox` objective (linear booster).

    Only single-record-per-subject (non time-varying) data is supported.

    Args:
        start (str or None): must be None; time-varying covariates are not
            supported.
        stop (str): name of the "stop"/event-time column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        covariates (list[str]): names of the covariate columns.
        dataset (pandas.DataFrame): the input dataset.
        device (str, optional): "cpu" or "cuda". Defaults to "cpu".

    Returns:
        tuple[list[float], float]: the fitted coefficients (in the same order as
            `covariates`, 0.0 for any covariate not used by the model) and the
            computation time in seconds.

    Raises:
        ValueError: if `start` is not None.
    """
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
    """Dispatches a Cox PH fit to the requested package's implementation.

    Args:
        package_name (str): one of "survival", "survivalgpu", "lifelines",
            "torchsurv", "xgboost".
        start (str or None): name of the "start" time column, or None for
            single-record-per-subject data.
        stop (str): name of the "stop"/event-time column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        covariates (list[str]): names of the covariate columns.
        dataset (pandas.DataFrame): the input dataset.
        ties (str): the ties handling method, e.g. "breslow" or "efron"
            ("torchsurv" always uses "efron").
        batch_size (int, optional): batch size, only used by "survivalgpu".
            Defaults to 0.
        n_bootstraps (int, optional): number of bootstrap resamples, only used
            by "survivalgpu". Defaults to 0.
        device (str, optional): "cpu" or "cuda". Defaults to "cpu".
        dtype (numpy.dtype, optional): floating point precision, only used by
            "survivalgpu". Defaults to np.float32.

    Returns:
        tuple[list[float], float]: the fitted coefficients (in the same order as
            `covariates`) and the computation time in seconds.

    Raises:
        ValueError: if `device` is "cuda" but CUDA is not available, if
            `device` is "cuda" but `package_name` does not support GPU
            (not in `CUDA_PACKAGES`), or if `package_name` is not recognized.
    """
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
    """Runs a Cox PH fit, optionally with bootstrapping, for benchmarking purposes.

    If `n_bootstraps > 0` and the package is "survivalgpu" running on "cuda",
    the bootstrap resamples are fit directly (a single timed call covering all
    bootstraps). Otherwise, a single fit without bootstraps is timed and, if
    `n_bootstraps > 0`, its computation time is extrapolated by multiplying by
    `n_bootstraps + 1`.

    Args:
        package_name (str): the package to use, see `run_package_cox`.
        start (str or None): name of the "start" time column, or None for
            single-record-per-subject data.
        stop (str): name of the "stop"/event-time column.
        event (str): name of the event indicator column (1 = event, 0 = censored).
        covariates (list[str]): names of the covariate columns.
        dataset (pandas.DataFrame): the input dataset.
        ties (str): the ties handling method, e.g. "breslow" or "efron".
        batch_size (int, optional): batch size, only used by "survivalgpu".
            Defaults to 0.
        n_bootstraps (int, optional): number of bootstrap resamples. Defaults
            to 0.
        device (str, optional): "cpu" or "cuda". Defaults to "cpu".
        dtype (numpy.dtype, optional): floating point precision, only used by
            "survivalgpu". Defaults to np.float32.

    Returns:
        tuple[list[float], float]: the fitted coefficients (in the same order as
            `covariates`) and the (possibly extrapolated) computation time in
            seconds.
    """
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



def run_cox_experiment(
    experiment_name,
    output_folder,
    n_patients_list,
    covariate_specs,
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
    covariate_specs : list of list of (kind, HR) tuples
        Each spec describes the covariates of one dataset, e.g.
        [("constant", 1.5), ("time_dep", 2.0)]. `kind` is either
        "constant" or "time_dep".
    package_device_dtype_list : list of (package_name, device, dtype) tuples
        E.g. [("survivalgpu", "cuda", np.float32), ("survival", "cpu", np.float64)]
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{experiment_name}.csv"

    df_results = []

    print(f"\n{'='*60}")
    print(f"Starting Cox experiment '{experiment_name}' | n_bootstraps={n_bootstraps} | n_iterations={n_iterations}")
    print(f"{'='*60}")

    print("\nWarming up packages with 500 patients...")
    warmup_dataset = simulate_dataset(
        max_time=max_time,
        n_patients=500,
        list_covariates=make_constant_covariates(1, [np.log(1.5)]),
        compress=True,
        seed=seed,
    )
    for package, device, dtype in package_device_dtype_list:
        print(f"  warming up {package} | dtype={dtype.__name__} | device={device}")
        run_benchmark_bootstraps(
            package_name = package,
            start        = None,
            stop         = "stop",
            event        = "events",
            covariates   = ["constant_1"],
            dataset      = warmup_dataset,
            ties         = "breslow",
            batch_size   = 0,
            n_bootstraps = 0,
            device       = device,
            dtype        = dtype,
        )

    for dataset_idx, (n_patients, covariate_spec) in enumerate(
        itertools.product(n_patients_list, covariate_specs)
    ):
        constant_betas = [np.log(hr) for kind, hr in covariate_spec if kind == "constant"]
        td_betas = [np.log(hr) for kind, hr in covariate_spec if kind == "time_dep"]
        n_constant = len(constant_betas)
        n_time_dep = len(td_betas)

        covariate_list = (
            make_constant_covariates(n_constant, constant_betas)
            + make_td_covariates(n_time_dep, td_betas)
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

        print(f"\n--- n_patients={n_patients} | n_constant={n_constant} | n_time_dep={n_time_dep} ---")
        for cov in covariate_list:
            print(f"  {cov.name}: beta={cov.coef:.4f}")
        print(f"  dataset: {len(dataset)} rows | simulation time: {t_sim:.4f} seconds")

        for package, device, dtype in package_device_dtype_list:
            print(f"  > {package} | dtype={dtype.__name__} | device={device} | n_iterations={n_iterations}")

            total_time = 0
            for i in range(n_iterations):
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
                print(f"      iteration {i+1}/{n_iterations}: {iter_time:.4f} seconds")

            computation_time = total_time / n_iterations
            print(f"    mean time over {n_iterations} iterations: {computation_time:.4f} seconds")

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


def build_covariates_with_betas(
    n_constant,
    n_time_dep,
    first_covariate_beta_list,
    other_covariates_beta_list,
):
    """Builds constant and time-dependent covariates with given coefficients.

    The first covariate's coefficient is taken from
    `first_covariate_beta_list[0]`; the remaining `n_constant + n_time_dep - 1`
    coefficients are taken from the start of `other_covariates_beta_list`.
    Constant covariates are assigned the first `n_constant` coefficients, and
    time-dependent covariates the remaining ones.

    Args:
        n_constant (int): the number of constant covariates to create.
        n_time_dep (int): the number of time-dependent covariates to create.
        first_covariate_beta_list (list[float]): a one-element list containing
            the coefficient for the first covariate.
        other_covariates_beta_list (list[float]): coefficients for the
            remaining covariates, must contain at least
            `n_constant + n_time_dep - 1` values.

    Returns:
        tuple[list, list[str], list[float]]: a tuple of
            (covariates, covariate_names, true_betas), where `covariates` is a
            list of `ConstantCovariate`/`TimeDependentCovariate` instances,
            `covariate_names` are their names, and `true_betas` are the
            coefficients used, in the same order as `covariates`.

    Raises:
        ValueError: if `n_constant + n_time_dep < 1`, if
            `first_covariate_beta_list` is empty, or if
            `other_covariates_beta_list` does not contain enough values.
    """
    total_cov = n_constant + n_time_dep
    if total_cov < 1:
        raise ValueError("At least one covariate is required.")

    if len(first_covariate_beta_list) < 1:
        raise ValueError("first_covariate_beta_list must contain at least 1 value.")

    expected_other_betas = total_cov - 1
    if len(other_covariates_beta_list) < expected_other_betas:
        raise ValueError(
            f"other_covariates_beta_list must contain at least {expected_other_betas} values "
            f"(got {len(other_covariates_beta_list)})."
        )

    beta_all = [first_covariate_beta_list[0]] + list(other_covariates_beta_list[:expected_other_betas])

    constant_betas = beta_all[:n_constant]
    td_betas = beta_all[n_constant:]

    covariates = make_constant_covariates(n_constant, constant_betas) + make_td_covariates(n_time_dep, td_betas)
    covariate_names = [c.name for c in covariates]
    true_betas = beta_all

    return covariates, covariate_names, true_betas


def validation_experiment(
    experiment_name,
    output_folder,
    n_patients_list,
    covariate_combination,
    beta_candidates,
    n_simulations=1,
    ties_list=("breslow", "efron"),
    max_time=365,
    seed=None,
    dtype=np.float32,
):
    """Compares `survivalgpu` and R's `survival` Cox coefficients on simulated datasets.

    For each combination of `n_patients_list` x `covariate_combination`, runs
    `n_simulations` simulations with randomly drawn true coefficients, fits a
    Cox PH model with both "survival" (R) and "survivalgpu" for each ties method
    in `ties_list`, and records the absolute and relative differences between
    the two packages' coefficients. Results (one row per covariate) are written
    incrementally to `<output_folder>/<experiment_name>.csv`. A per-combination
    summary, aggregating the per-simulation mean differences, is written to
    `<output_folder>/<experiment_name>_summary.csv`.

    Args:
        experiment_name (str): name used for the output file names.
        output_folder (str or Path): directory where results are written
            (created if needed).
        n_patients_list (list[int]): numbers of patients to simulate.
        covariate_combination (list[tuple[int, int]]): list of
            (n_constant, n_time_dep) covariate-count combinations.
        beta_candidates (list[float]): candidate true regression coefficients
            (log-hazard-ratio scale) drawn at random for each covariate of
            each simulation.
        n_simulations (int, optional): number of simulations per combination.
            Defaults to 1.
        ties_list (str or tuple[str], optional): ties handling method(s) to
            evaluate, e.g. "breslow" and/or "efron". Defaults to
            ("breslow", "efron").
        max_time (int, optional): maximum follow-up time for the simulated
            datasets. Defaults to 365.
        seed (int, optional): random seed, for reproducibility. Defaults to None.
        dtype (numpy.dtype, optional): floating point precision used by
            `survivalgpu`. Defaults to np.float32.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: the detailed per-covariate
            results, and the per-combination summary (mean/median/max of the
            mean absolute and relative differences across simulations).
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{experiment_name}.csv"

    print(f"\n{'=' * 60}")
    print(f"Starting Cox validation experiment '{experiment_name}' | n_simulations={n_simulations} | ties_list={ties_list}")
    print(f"{'=' * 60}")

    rows = []

    if isinstance(ties_list, str):
        ties_list = [ties_list]
    ties_list = list(ties_list)

    rng = np.random.default_rng(seed=seed)

    combo_iter = itertools.product(n_patients_list, covariate_combination)
    for combo_idx, (n_patients, (n_constant, n_time_dep)) in enumerate(combo_iter):
        total_cov = n_constant + n_time_dep
        print(f"\n{'=' * 60}")
        print(f"Running n_patients={n_patients}, n_constant={n_constant}, n_time_dep={n_time_dep}")
        print(f"{'=' * 60}")

        for sim in range(n_simulations):
            sim_start = time.time()
            random_betas = rng.choice(beta_candidates, size=total_cov, replace=True).tolist()
            print(f"\n--- Simulation {sim + 1}/{n_simulations} | true_betas={[round(b, 4) for b in random_betas]} ---")

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

                print(f"  ties={ties:<8} | mean_abs_diff={mean_abs_diff:.3e} | mean_rel_diff={mean_rel_diff:.3e}")

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

            print(f"  simulation time: {time.time() - sim_start:.4f} seconds")

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
