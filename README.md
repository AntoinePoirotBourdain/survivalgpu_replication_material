# survivalGPU — Replication material

Code to reproduce the figures, tables and benchmarks of the survivalGPU
paper: Cox and WCE (Weighted Cumulative Exposure) model fits compared
against reference R/Python implementations, with and without GPU
acceleration.

## Installation

The `survivalgpu` package lives in the sibling `survivalGPU` repository.
Clone it next to this one, so that `survivalGPU/` and
`survivalgpu_replication_material/` are sibling directories.

### Python

The code requires Python >= 3.10. From this directory
(`survivalgpu_replication_material`), create and activate a virtual
environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

Then install survivalGPU (with the `all` extra) and this repository:

```bash
pip install -e ../survivalGPU[all]
pip install -e .
```

`pykeops`, one of survivalGPU's core dependencies, just-in-time compiles
CUDA/C++ kernels, so a working C++ compiler is required; a CUDA-capable GPU
is optional (the package falls back to CPU otherwise).

### R

The R-only `illustration.R` script (see below) and the WCE
validation/benchmark code (called via `rpy2`) need R, plus the `WCE`,
`remotes`, and `reticulate` packages:

```r
install.packages(c("WCE", "remotes", "reticulate"))
```

The survivalGPU R package is a thin wrapper around the Python package
installed above, calling into it via `reticulate`. Install it from the
sibling clone:

```r
remotes::install_local("../survivalGPU/R")
```

By default, `reticulate` picks its own Python interpreter, which may not
have `survivalgpu` installed. Point it at the virtual environment created
in the Python setup step by setting `RETICULATE_PYTHON` before launching R:

```bash
export RETICULATE_PYTHON=/path/to/survivalgpu_replication_material/.venv/bin/python
```

> **Warning**: survivalGPU is not currently available on Windows.

## Reproducing the paper's figures and tables

All commands below assume the working directory is
`survivalgpu_replication_material` (this repository's root).

### 1. Run the benchmark / validation experiments

```bash
bash scripts/figure2-4.sh   # Cox benchmark (figures 2-4)      — requires a CUDA GPU
bash scripts/figure5.sh     # WCE benchmark without bootstraps (figure 5) — requires a CUDA GPU
bash scripts/figure6.sh     # WCE benchmark with bootstraps (figure 6)    — requires a CUDA GPU
bash scripts/table1-4.sh    # Cox and WCE validation (tables 1-4)
```

`figure2-4.sh`, `figure5.sh`, and `figure6.sh` benchmark `survivalgpu` on
`device="cuda"` and therefore need a CUDA-capable GPU. `table1-4.sh` falls
back to the CPU automatically when no GPU is available (see
`wce_functions.py`), so it can be run on any machine.

Each one runs in the background and logs to `logs/log_<name>.txt`; raw
results and their summaries (median/max relative differences) are written
as CSV to `results/` (`figure5.sh` writes to `benchmark_results/`). Tail a
log to follow progress, e.g.:

```bash
tail -f logs/log_figure2-4.txt
```

`table1-4.sh` produces four validation tables: table 1 (Cox, float32),
table 2 (Cox, float64), table 3 (WCE, float32), and table 4 (WCE,
float64). In the paper itself, tables 1 and 2 are combined into a single
table 1, and tables 3 and 4 are combined into a single table 2 — the
float32/float64 columns sit side by side rather than in separate tables.

Approximate wall-clock time on our machine (single CUDA GPU), from the logs
in `logs/`:

| Script | Experiment | Time |
| --- | --- | --- |
| `figure2-4.sh` | figure 2 | 40m 22s |
| | figure 3 | 9m 16s |
| | figure 4 | 1h 9m 55s |
| | **total** | **~1h 59m** |
| `figure5.sh` | figure 5 | ~7h 55m |
| `figure6.sh` | figure 6 | ~16h 5m |
| `table1-4.sh` | table 1 | 13m 8s |
| | table 2 | 12m 53s |
| | table 3 | 46m 47s |
| | table 4 | 46m 52s |
| | **total** | **~1h 59m** |

### 2. Generate the figures (PDFs) from the results

```bash
bash scripts/generate_images.sh   # figures 2-6 and 8 (WCE weight scenarios)
```

PDFs are written to `figures/`.

### 3. Worked example ("Illustration" section, figure 7)

`illustration.R` reproduces the `drugdata` worked example from the paper's
Illustration section: a BIC-based model selection grid over `nknots` and
`constrained`, a bootstrap refit (`B = 100`) of the selected model, the
weight-function plot, and the hazard-ratio calculation. Unlike the rest of
this repository, it drives survivalGPU's R interface directly rather than
its Python API. It needs the R setup and `RETICULATE_PYTHON` environment
variable described in "Installation" above.

With `RETICULATE_PYTHON` set and the working directory set to this
repository (`survivalgpu_replication_material`):

```bash
Rscript illustration.R
```

or, from an interactive R session:

```r
setwd("path/to/survivalgpu_replication_material")
source("illustration.R")
```

The script attaches `WCE` and `survivalGPU`, loads `WCE::drugdata`, and runs
the workflow above. It prints the BIC comparison table, the selected
model's `summary()` output, and the hazard-ratio estimate to the console,
and writes the weight-function plot to `figures/figure_7.pdf`.

## Repository layout

- `cox_functions.py`, `wce_functions.py`, `simulation_functions.py` — shared
  helpers (model-fitting wrappers, simulated dataset generation) used by the
  experiment scripts.
- `figure_generation/` — scripts that run the Cox/WCE benchmark and
  validation experiments (figures 2-6, tables 1-4; see note above on how
  these map to the paper's tables 1-2).
- `image_generation/` — scripts that turn experiment results (`results/`,
  `benchmark_results/`) into the paper's PDF figures (`figures/`).
- `scripts/` — shell wrappers around the experiment scripts.
- `illustration.R` — standalone R script reproducing the paper's
  "Illustration" section (figure 7) on the `WCE::drugdata` data set.
- `results/`, `benchmark_results/` — experiment output (CSV).
- `figures/` — generated PDF figures.
- `logs/` — logs from background experiment runs.
