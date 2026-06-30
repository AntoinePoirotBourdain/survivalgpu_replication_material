# survivalGPU — Replication material

Code to reproduce the figures, tables and benchmarks of the survivalGPU
paper: Cox and WCE (Weighted Cumulative Exposure) model fits compared
against reference R/Python implementations, with and without GPU
acceleration.

## Setup

The code requires Python >= 3.10, and the `survivalgpu` package, which lives
in the sibling `survivalGPU` repository. Clone it next to this one, so that
`survivalGPU/` and `survivalgpu_replication_material/` are sibling
directories.

From this directory (`survivalgpu_replication_material`), create and
activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

Then install survivalGPU (with the `all` extra) and this repository:

```bash
pip install -e ../survivalGPU[all]
pip install -e .
```

The WCE validation/benchmark code calls R via `rpy2`. Install R, then the
`survival` and `WCE` packages from an R session:

```r
install.packages(c("survival", "WCE"))
```

## Reproducing the paper's figures and tables

All commands below assume the working directory is
`survivalgpu_replication_material` (this repository's root).

### 1. Run the benchmark / validation experiments

```bash
bash scripts/figure2-4.sh   # Cox benchmark (figures 2-4)
bash scripts/figure5.sh     # WCE benchmark without bootstraps (figure 5)
bash scripts/figure6.sh     # WCE benchmark with bootstraps (figure 6)
bash scripts/table1.sh      # Cox validation (table 1)
bash scripts/table2.sh      # WCE validation (table 2)
```

Each one runs in the background and logs to `logs/log_<name>.txt`; raw
results are written as CSV to `results/` (`figure5.sh` writes to
`benchmark_results/`). Tail a log to follow progress, e.g.:

```bash
tail -f logs/log_figure2-4.txt
```

### 2. Generate the figures (PDFs) from the results

```bash
bash scripts/generate_images.sh                 # figures 2-6
python image_generation/generate_image_8.py     # figure 8 (WCE weight scenarios)
```

PDFs are written to `figures/`.

## Repository layout

- `cox_functions.py`, `wce_functions.py`, `simulation_functions.py` — shared
  helpers (model-fitting wrappers, simulated dataset generation) used by the
  experiment scripts.
- `figure_generation/` — scripts that run the Cox/WCE benchmark and
  validation experiments (figures 2-6, tables 1-2).
- `image_generation/` — scripts that turn experiment results (`results/`,
  `benchmark_results/`) into the paper's PDF figures (`figures/`).
- `scripts/` — shell wrappers around the experiment scripts.
- `results/`, `benchmark_results/` — experiment output (CSV).
- `figures/` — generated PDF figures.
- `logs/` — logs from background experiment runs.
