
import numpy as np
import pandas as pd

from survivalgpu import simulate_dataset, ConstantCovariate, TimeDependentCovariate

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
    