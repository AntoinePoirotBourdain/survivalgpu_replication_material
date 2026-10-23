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

Then install survivalGPU (with the `wce` extra, which adds `pykeops`) and
this repository:

```bash
pip install -e "../survivalGPU[wce]"
pip install -e .
```

`pykeops`, needed by the WCE model, just-in-time compiles CUDA/C++ kernels,
so a working C++ compiler is required; a CUDA-capable GPU is optional (the
package falls back to CPU otherwise).

### R

The two R illustration scripts (see below) and the validation/benchmark
code that calls R through `rpy2` need R >= 4.1 and these packages:

| Package | Used by |
| --- | --- |
| `survival` | reference Cox fits (validation), `illustration_cox.R` |
| `WCE` | reference WCE fits (validation/benchmarks), `drugdata` in `illustration_wce.R` |
| `survminer` | plots in `illustration_cox.R` |
| `reticulate` | used by the survivalGPU R package to call Python |
| `remotes` | installing the survivalGPU R package below |

```r
install.packages(c("survival", "WCE", "survminer", "reticulate", "remotes"))
```

The survivalGPU R package is a thin wrapper around the Python package
installed above, calling into it via `reticulate`. Install it from the
sibling clone:

```r
remotes::install_local("../survivalGPU/R")
```

By default, `reticulate` picks its own Python interpreter, which may not
have `survivalgpu` installed. Point it at the virtual environment created
in the Python setup step by setting `RETICULATE_PYTHON` before launching R
(from this directory):

```bash
export RETICULATE_PYTHON="$PWD/.venv/bin/python"
```

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

### 3. Illustration (Section 7 of the paper)

`illustration_cox.R` and `illustration_wce.R` reproduce Section 7
(Illustration) of the paper, for the Cox and WCE models. With
`RETICULATE_PYTHON` set (see "Installation"):

```bash
Rscript illustration_cox.R
Rscript illustration_wce.R
```

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
- `illustration_cox.R`, `illustration_wce.R` — Section 7 (Illustration) of
  the paper, for the Cox and WCE models.
- `results/`, `benchmark_results/` — experiment output (CSV).
- `figures/` — generated PDF figures.
- `logs/` — logs from background experiment runs.
