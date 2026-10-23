# ============================================================
# Illustration section of the survivalGPU JSS paper -- WCE model.
#
# Run with: Rscript illustration_wce.R
# (requires RETICULATE_PYTHON, see the README)
# ============================================================

library("WCE")
library("survivalGPU")

data("drugdata", package = "WCE")
print(head(drugdata))


cutoff <- 180

fit <- wceGPU(
  data = drugdata, id = "Id", event = "Event", start = "Start",
  stop = "Stop", expos = "dose", covariates = c("age", "sex"),
  nknots = 1:3, cutoff = cutoff, constrained = "Right", x = TRUE, device = "cpu"
)

summary(fit)

fit_unconstrained <- wceGPU(
  data = drugdata, id = "Id", event = "Event", start = "Start",
  stop = "Stop", expos = "dose", covariates = c("age", "sex"),
  nknots = 1:3, cutoff = cutoff, constrained = "None", x = TRUE, device = "cpu"
)

summary(fit_unconstrained)


print(coef(fit))

pdf("figures/figure_8.pdf", width = 6, height = 5)
plot(fit)
invisible(dev.off())

exposed   <- rep(1, cutoff)
unexposed <- rep(0, cutoff)
print(HR(fit, exposed, unexposed))


fit_boot <- bootstrap(fit, R = 100, batchsize = 25)
print(HR(fit_boot, exposed, unexposed))


pdf("figures/figure_9.pdf", width = 6, height = 5)
plot(fit_boot)
invisible(dev.off())






# # #####


# # invisible(reticulate::import("torch")$manual_seed(123L))


# pdf("figures/figure_8.pdf", width = 6, height = 5)
# plot(result_boot)
# invisible(dev.off())


# exposed   <- rep(1, cutoff)
# unexposed <- rep(0, cutoff)

# print(HR(result_boot, exposed, unexposed))
