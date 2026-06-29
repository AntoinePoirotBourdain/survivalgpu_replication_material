from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

############ Figure: Benchmark of Cox
data_path = Path("results")
df = pd.read_csv(data_path / "figure_2_cox.csv")

df = df[df["n_patients"] >= 10000]

r_survival       = df[df["package"] == "cox_r_survival"]
lifelines        = df[df["package"] == "cox_lifelines"]
survivalgpu_cpu  = df[df["package"] == "cox_survival_gpu_cpu_f32"]
survivalgpu_cuda = df[df["package"] == "cox_survival_gpu_cuda_f32"]

n_patients            = survivalgpu_cuda["n_patients"].values
r_survival_time       = r_survival["mean_time_s"].values
lifelines_time        = lifelines["mean_time_s"].values
survivalgpu_cpu_time  = survivalgpu_cpu["mean_time_s"].values
survivalgpu_cuda_time = survivalgpu_cuda["mean_time_s"].values

speedup_against_r_survival      = r_survival_time      / survivalgpu_cuda_time
speedup_against_lifelines       = lifelines_time        / survivalgpu_cuda_time
speedup_against_survivalgpu_cpu = survivalgpu_cpu_time  / survivalgpu_cuda_time

# --- Style ---
survivalgpu_style     = {"color": "#0072B2", "marker": "D", "ls": "-",  "linewidth": 2}
survivalgpu_cpu_style = {"color": "#0072B2", "marker": "x", "ls": "--", "linewidth": 1.5}
survival_style        = {"color": "#000000", "marker": "s", "ls": ":",  "linewidth": 1.5}
lifelines_style       = {"color": "#E69F00", "marker": "v", "ls": ":",  "linewidth": 1.5}

plt.close("all")
fig, ax = plt.subplots(ncols=2, figsize=(14, 5))

# --- Computation time ---
ax[0].plot(n_patients, r_survival_time,       label="R survival",         **survival_style)
ax[0].plot(n_patients, lifelines_time,        label="lifelines",          **lifelines_style)
ax[0].plot(n_patients, survivalgpu_cpu_time,  label="survivalGPU (CPU)",  **survivalgpu_cpu_style)
ax[0].plot(n_patients, survivalgpu_cuda_time, label="survivalGPU",        **survivalgpu_style)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Number of patients")
ax[0].set_ylabel("Computation time (s)")
ax[0].set_title("Computation time")
ax[0].legend(frameon=False, fontsize=9)
ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax[0].grid(True, which="major", ls="-",  lw=0.5, alpha=0.4)
ax[0].grid(True, which="minor", ls="--", lw=0.3, alpha=0.3)

# --- Speedup ---
ax[1].plot(n_patients, speedup_against_r_survival,      label="Against R survival",        **survival_style)
ax[1].plot(n_patients, speedup_against_lifelines,       label="Against lifelines",         **lifelines_style)
ax[1].plot(n_patients, speedup_against_survivalgpu_cpu, label="Against survivalGPU (CPU)", **survivalgpu_cpu_style)

ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel("Number of patients")
ax[1].set_ylabel("Speedup")
ax[1].set_title("Speedup of the survivalGPU implementation")
ax[1].legend(frameon=False, fontsize=9)
ax[1].set_ylim([0.9, 100])
ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}x"))
ax[1].grid(True, which="major", ls="-",  lw=0.5, alpha=0.4)
ax[1].grid(True, which="minor", ls="--", lw=0.3, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/figure_2.pdf", bbox_inches="tight")
plt.close(fig)