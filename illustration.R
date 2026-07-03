# ============================================================
# Illustration section of the survivalGPU JSS paper.
#
# Worked example on WCE::drugdata: BIC-based model selection over a
# grid of (nknots, constrained), bootstrap refit of the selected
# model, weight-function plot (figures/figure_7.pdf), and hazard
# ratio for a constant-exposure vs. unexposed scenario.
#
# Run with: Rscript illustration.R
# (requires RETICULATE_PYTHON set to the Python venv where survivalgpu is
# installed — see the "Installation" section of this repository's README)
# ============================================================

library("WCE")
library("survivalGPU")

data("drugdata", package = "WCE")

cutoff <- 180

# ---- Model selection: BIC over a grid of nknots x constrained ----
# wceGPU() reports the BIC by default (aic = FALSE); constrained = FALSE
# corresponds to the unconstrained ("None") weight function.

# expand.grid() varies its first argument fastest, so listing "constrained"
# first keeps rows in (nknots, constrained) order: (1, Right), (1, None),
# (2, Right), (2, None), (3, Right), (3, None) -- matching the paper's table.
grid <- expand.grid(
  constrained = c("Right", "None"),
  nknots      = 1:3,
  stringsAsFactors = FALSE
)[, c("nknots", "constrained")]

grid$bic <- vapply(seq_len(nrow(grid)), function(i) {
  constrained_arg <- if (grid$constrained[i] == "None") FALSE else grid$constrained[i]

  fit <- wceGPU(
    data = drugdata, id = "Id", event = "Event", start = "Start",
    stop = "Stop", expos = "dose", covariates = c("age", "sex"),
    nknots = grid$nknots[i], cutoff = cutoff,
    constrained = constrained_arg
  )

  fit$info.criterion[1]
}, numeric(1))

print(grid, row.names = FALSE)

best <- grid[which.min(grid$bic), ]

# ---- Refit the selected model, with bootstrap CIs ----

set.seed(123)
model <- wceGPU(
  data = drugdata, id = "Id", event = "Event", start = "Start",
  stop = "Stop", expos = "dose", covariates = c("age", "sex"),
  nknots = best$nknots, cutoff = cutoff,
  constrained = if (best$constrained == "None") FALSE else best$constrained,
  nbootstraps = 100
)

summary(model)

# ---- Weight-function plot ----

pdf("figures/figure_7.pdf", width = 6, height = 5)
plot(model)
dev.off()

# ---- Hazard ratio: constant exposure vs. unexposed over the cutoff window ----

exposed   <- rep(1, cutoff)
unexposed <- rep(0, cutoff)

hr <- HR(model, exposed, unexposed)
print(hr)