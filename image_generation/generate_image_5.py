from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

############ Figure 5: Benchmark of WCE
data_path = Path("results")
df = pd.read_csv(data_path / "figure_5_wce.csv")

df = df[df["n_patients"] >= 1000]

survivalgpu_cuda = df[(df["package"] == "survivalgpu") & (df["device"] == "cuda")]
survivalgpu_cpu  = df[(df["package"] == "survivalgpu") & (df["device"] == "cpu")]
wce_cpu          = df[(df["package"] == "WCE")         & (df["device"] == "cpu")]

n_patients                      = survivalgpu_cuda["n_patients"].values
survivalgpu_cuda_time           = survivalgpu_cuda["computation_time"].values
survivalgpu_cpu_time            = survivalgpu_cpu["computation_time"].values
wce_cpu_time                    = wce_cpu["computation_time"].values
speedup_against_wce_cpu         = wce_cpu_time / survivalgpu_cuda_time
speedup_against_survivalgpu_cpu = survivalgpu_cpu_time / survivalgpu_cuda_time

# --- Style ---
survivalgpu_style     = {"color": "#0072B2", "marker": "D", "ls": "-",  "linewidth": 2}
survivalgpu_cpu_style = {"color": "#0072B2", "marker": "x", "ls": "--", "linewidth": 1.5}
wce_style             = {"color": "#000000", "marker": "s", "ls": ":",  "linewidth": 1.5}

plt.close("all")
fig, ax = plt.subplots(ncols=2, figsize=(14, 5))

# --- Computation time ---
ax[0].plot(n_patients, wce_cpu_time,          label="WCE (CPU)",         **wce_style)
ax[0].plot(n_patients, survivalgpu_cpu_time,  label="survivalGPU (CPU)", **survivalgpu_cpu_style)
ax[0].plot(n_patients, survivalgpu_cuda_time, label="survivalGPU",       **survivalgpu_style)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Number of patients")
ax[0].set_ylabel("Computation time (s)")
ax[0].set_title("Computation time")
ax[0].legend()
ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax[0].grid(True, which="major", ls="-",  lw=0.5, alpha=0.4)
ax[0].grid(True, which="minor", ls="--", lw=0.3, alpha=0.3)

# --- Speedup ---
ax[1].plot(n_patients, speedup_against_wce_cpu,         label="Against WCE (CPU)",         **wce_style)
ax[1].plot(n_patients, speedup_against_survivalgpu_cpu, label="Against survivalGPU (CPU)", **survivalgpu_cpu_style)

ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel("Number of patients")
ax[1].set_ylabel("Speedup")
ax[1].set_title("Speedup of the survivalGPU implementation")
ax[1].legend()
ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}x"))
ax[1].grid(True, which="major", ls="-",  lw=0.5, alpha=0.4)
ax[1].grid(True, which="minor", ls="--", lw=0.3, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/figure_5.pdf", bbox_inches="tight")
plt.close(fig)