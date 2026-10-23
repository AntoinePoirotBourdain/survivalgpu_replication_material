############### Cox Model

library("survival")
library("dplyr")

data("drugdata",package="WCE")

compressed_drugdata <- survcondense(
  Surv(Start, Stop, Event) ~ age + sex + dose,
  data = drugdata,
  id   = Id
)
compressed_drugdata <- relocate(compressed_drugdata, Id)

print(head(compressed_drugdata))

library("survivalGPU")

result <- coxphGPU(Surv(Start, Stop, Event) ~ dose + age + sex,
                data = compressed_drugdata, 
                x = TRUE, patient_id = "Id", device ="cpu")

# the seed for the bootstraps must be done in numpy 

invisible(reticulate::import("torch")$manual_seed(123L))

result_boot <- bootstrap(result, R = 100, batchsize = 25)

summary(result_boot)

# summary(result_boot)

zph <- cox.zph(result)
print(zph)


library("survminer")



pdf("figures/figure_7.pdf", width = 6, height = 5)
print(ggadjustedcurves(result, variable = "sex", data = drugdata))
invisible(dev.off())

zph <- cox.zph(result)
print(zph)






# pdf("figures/illustration_cox_curves.pdf", width = 6, height = 5)
# print(zph)
# print(ggcoxzph(zph))






# # ---- Cox model: current dose as a time-dependent covariate ----

# fit <- coxphGPU(Surv(Start, Stop, Event) ~ dose + age + sex,
#                 data = drugdata, x = TRUE, patient_id = "Id")

# # ---- Bootstrap (resampling of patients) ----

# # The resamples are drawn by PyTorch: set.seed() has no effect on them.
# invisible(reticulate::import("torch")$manual_seed(123L))
# fit_boot <- bootstrap(fit, R = 1000)

# summary(fit_boot)

# # ---- Adjusted survival curves, by sex ----

# pdf("figures/illustration_cox_curves.pdf", width = 6, height = 5)
# print(ggadjustedcurves(fit_boot, variable = "sex", data = drugdata))
# invisible(dev.off())

# # ---- Proportional hazards assumption (Schoenfeld residuals) ----

# zph <- cox.zph(fit_boot)
# print(zph)

# pdf("figures/illustration_cox_zph.pdf", width = 6, height = 10)
# print(ggcoxzph(zph))
# invisible(dev.off())

# # ---- Forest plot ----
# # Confidence intervals from the normal approximation, not the bootstrap.

# pdf("figures/illustration_cox_forest.pdf", width = 7, height = 4)
# print(ggforest(fit_boot, data = drugdata))
# invisible(dev.off())
