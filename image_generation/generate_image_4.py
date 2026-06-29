from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

############ Figure: Benchmark of survivalGPU vs xgboost vs torchsurv
data_path = Path("results")
df = pd.read_csv(data_path / "figure_4_gpu_cox.csv")

df = df[df["n_patients"] >= 5000]

survivalgpu = df[df["package"] == "survivalgpu"]
xgboost     = df[df["package"] == "xgboost"]
torchsurv   = df[df["package"] == "torchsurv"]

n_patients        = survivalgpu["n_patients"].values
survivalgpu_time  = survivalgpu["computation_time"].values
xgboost_time      = xgboost["computation_time"].values
torchsurv_time    = torchsurv["computation_time"].values

# --- Style ---
# --- Style ---
# --- Style ---
# --- Style ---
survivalgpu_style = {"color": "#0072B2", "marker": "D", "ls": "-",  "linewidth": 2}
xgboost_style     = {"color": "#E69F00", "marker": "o", "ls": ":",  "linewidth": 1.5}
torchsurv_style   = {"color": "#009E73", "marker": "v", "ls": ":",  "linewidth": 1.5}

plt.close("all")
fig, ax = plt.subplots(ncols=1, figsize=(7, 5))

# --- Computation time ---
ax.plot(n_patients, xgboost_time,     label="xgboost",     **xgboost_style)
ax.plot(n_patients, torchsurv_time,   label="torchsurv",   **torchsurv_style)
ax.plot(n_patients, survivalgpu_time, label="survivalGPU", **survivalgpu_style)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of patients")
ax.set_ylabel("Computation time (s)")
ax.set_title("Computation time")
ax.legend(frameon=False, fontsize=9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, which="major", ls="-",  lw=0.5, alpha=0.4)
ax.grid(True, which="minor", ls="--", lw=0.3, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/figure_4.pdf", bbox_inches="tight")
plt.close(fig)