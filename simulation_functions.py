
import time

import numpy as np
import pandas as pd

from survivalgpu import simulate_dataset, ConstantCovariate, TimeDependentCovariate


def simulate_dataset_in_batches(max_time, n_patients, list_covariates, batch_size=50000, compress=True, seed=None):
    """Simulates a dataset of `n_patients` patients in batches of `batch_size`.

    Simulating very large numbers of patients in a single `simulate_dataset`
    call can be slow. This splits the work into successive batches, offsetting
    each batch's "patients" column so that patient ids stay unique and
    contiguous across the full dataset.

    Callers are expected to only use this once `n_patients` is large enough
    that batching is worth it (see e.g. `run_cox_experiment`, which only
    calls this for `n_patients >= 100000`).

    Args:
        max_time (int): the maximum follow-up time, passed to `simulate_dataset`.
        n_patients (int): the total number of patients to simulate.
        list_covariates (list): the covariates to simulate, passed to `simulate_dataset`.
        batch_size (int, optional): number of patients simulated per batch. Defaults to 50000.
        compress (bool, optional): passed to `simulate_dataset`. Defaults to True.
        seed (int, optional): random seed for the first batch; subsequent batches
            use `seed + 1000 * i`, to keep batch seeds well clear of any other
            seed already offset by the caller. Defaults to None.

    Returns:
        pandas.DataFrame: the concatenated dataset, with `n_patients` distinct,
            contiguous patient ids.
    """
    n_full_batches, remainder = divmod(n_patients, batch_size)
    batch_sizes = [batch_size] * n_full_batches
    if remainder > 0:
        batch_sizes.append(remainder)
    n_batches = len(batch_sizes)

    if n_batches > 1:
        print(f"Generating {n_patients} patients in {n_batches} batches of up to {batch_size}...")

    t_total_start = time.time()
    batches = []
    patient_offset = 0
    for i, size in enumerate(batch_sizes):
        t_batch_start = time.time()
        batch_seed = None if seed is None else seed + 1000 * i
        batch = simulate_dataset(
            max_time=max_time,
            n_patients=size,
            list_covariates=list_covariates,
            compress=compress,
            seed=batch_seed,
        )
        batch["patients"] += patient_offset
        patient_offset += size
        batches.append(batch)

        if n_batches > 1:
            print(f"  batch {i + 1}/{n_batches} ({size} patients): {time.time() - t_batch_start:.2f}s")

    dataset = pd.concat(batches, ignore_index=True) if len(batches) > 1 else batches[0]

    if n_batches > 1:
        print(f"Total simulation time for {n_patients} patients: {time.time() - t_total_start:.2f}s")

    return dataset


def quick_simulated_dataset(n_patients, n_constant_covariates, n_time_dependent_covariates, beta_list, max_time):

    covariates = []
    for i in range(n_constant_covariates):
        w = 0.5#np.random.uniform(0.2, 0.8)
        covariates.append(ConstantCovariate(
            name=f"constant_{i + 1}",
            coef=float(np.random.choice(beta_list)),
            values=[0, 1],
            weights=[w, 1 - w],
        ))
    for j in range(n_time_dependent_covariates):
        covariates.append(TimeDependentCovariate(
            name=f"time_dep_{j + 1}",
            values=[0.5, 1.0, 1.5, 2.0],
            coef=float(np.random.choice(beta_list)),
        ))

    return simulate_dataset(
        max_time=max_time,
        n_patients=n_patients,
        list_covariates=covariates,
        compress=False,
    )
    