# survivalGPU — Replication material

Code to reproduce the figures, tables and benchmarks of the survivalGPU
paper: Cox and WCE (Weighted Cumulative Exposure) model fits compared
against reference R/Python implementations, with and without GPU
acceleration.

## Installation

The Python package `survivalgpu` and the R package `survivalGPU` are
installed from their source packages, `survivalgpu-0.2.0.tar.gz` and
`survivalGPU_0.1.0.tar.gz`, provided separately from this replication
material. Copy both files into this directory
(`survivalgpu_replication_material`) before running the commands below.

### Python

The code requires Python >= 3.10. From this directory
(`survivalgpu_replication_material`), create and activate a virtual
environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

Install `survivalgpu` from its source package, with the `wce` extra (which
adds `pykeops`), then the code of this directory and its other
dependencies, listed in `pyproject.toml`:

```bash
pip install "./survivalgpu-0.2.0.tar.gz[wce]"
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
| `survminer` | plot in `illustration_cox.R` |
| `dplyr` | `illustration_cox.R` |
| `reticulate` | used by the survivalGPU R package to call Python |

```r
install.packages(c("survival", "WCE", "survminer", "dplyr", "reticulate"))
```

Then install the survivalGPU R package from its source package, from this
directory:

```r
install.packages("survivalGPU_0.1.0.tar.gz", repos = NULL, type = "source")
```

The R package includes its own copy of the Python code of `survivalgpu`,
and calls it through `reticulate`, with the Python packages (`torch`,
`pykeops`, ...) of the virtual environment created above. By default,
`reticulate` picks its own Python interpreter, which may not have these
packages. Point it at that virtual environment by setting
`RETICULATE_PYTHON` before launching R (from this directory):

```bash
export RETICULATE_PYTHON="$PWD/.venv/bin/python"
```

## Computing environment

The results in `results/`, `figures/` and `logs/`, and the timings below,
were obtained on this machine:

| | |
| --- | --- |
| CPU | Intel Xeon W-2255 (3.70 GHz) |
| GPU | NVIDIA GeForce RTX 3090 (24 GB), driver 535.274.02 |
| OS | Linux 6.2 (host), Ubuntu 24.04 (container) |

The code ran in a Singularity container built from the KeOps Docker image
`getkeops/keops-full:2.3-geomloss0.2.6-cuda12.8.1-pytorch2.8.0-python3.13`
(<https://github.com/getkeops/keops>):

```bash
singularity build keops-full.sif docker://getkeops/keops-full:2.3-geomloss0.2.6-cuda12.8.1-pytorch2.8.0-python3.13
```

This image provides Python 3.13.0 with `torch` 2.8.0 (CUDA 12.8), `numpy`
2.3.1, `pandas` 2.3.2, `matplotlib` 3.10.0, `pykeops` 2.3 and `rpy2` 3.6.2,
and R 4.3.3 with `dplyr` 1.1.4. The other packages were installed in the
container:

| | Packages added to the image |
| --- | --- |
| Python | `survivalgpu` 0.1.0, `lifelines` 0.30.3, `torchsurv` 0.1.6, `xgboost` 3.2.0 |
| R | `survival` 3.8-6, `WCE` 1.0.4, `reticulate` 1.46.0, `survminer` |

`survival`, `WCE` and `reticulate` are also in the image, in older versions
(3.5-8, 1.0.3 and 1.35.0); the versions above were installed over them.

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
