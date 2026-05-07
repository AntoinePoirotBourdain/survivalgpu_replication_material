
import pandas as pd
df = pd.read_csv("benchmark_results/val_breslow_efron.csv")

print(df.head())

df_summary = (
    df.drop_duplicates(
        subset=["simulation", "ties", "n_patients", "n_constant_covariates", "n_time_dependent_covariates"]
    )
    .groupby(["ties", "n_patients", "n_constant_covariates", "n_time_dependent_covariates"], as_index=False)
    .agg(
        mean_of_mean_abs_diff=("mean_abs_diff", "mean"),
        median_of_mean_abs_diff=("mean_abs_diff", "median"),
        std_of_mean_abs_diff=("mean_abs_diff", "std"),
        mean_of_mean_rel_diff=("mean_rel_diff", "mean"),
        median_of_mean_rel_diff=("mean_rel_diff", "median"),
        std_of_mean_rel_diff=("mean_rel_diff", "std"),
        n_simulations=("simulation", "count"),
    )
)

print(df_summary)
df_summary.to_csv("benchmark_results/val_breslow_efron_summary.csv", index=False)