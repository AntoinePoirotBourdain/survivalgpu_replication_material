from pathlib import Path
import matplotlib.pyplot as plt

from survivalgpu.simulations import get_scenario

############ Figure 8: WCE weight scenarios
max_time = 365

scenarios = [
    ("exponential_scenario", "Exponential scenario", r"$w(u) = 7 \cdot e^{-7u/365}$"),
    ("bi_linear_scenario",   "Bilinear scenario",     r"$w(u) = (1 - u/50)$ if $u < 50$ then $0$"),
    ("early_peak_scenario",  "Early Peak scenario",   r"$w(u) = \phi(u/365;\ \mu=0.04,\ \sigma=0.05)$"),
    ("inverted_u_scenario",  "Inverted U scenario",   r"$w(u) = \phi(u/365;\ \mu=0.2,\ \sigma=0.06)$"),
    ("constant_scenario",    "Constant scenario",     r"$w(u) = 1$ if $u \leq 180$ then $0$"),
    ("hat_scenario",         "Hat scenario",          r"$w(u) = (u/180)$ then $[1-(u-180)/60]$"),
]

plt.close("all")
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

for ax, (scenario_name, title, formula) in zip(axes.flat, scenarios):
    weights = get_scenario(scenario_name, max_time)
    ax.plot(range(max_time), weights, color="black", linewidth=2)

    ax.set_title(f"{title}\n{formula}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized Weight")
    ax.grid(True, ls="-", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig("figures/figure_8.pdf", bbox_inches="tight")
plt.close(fig)
