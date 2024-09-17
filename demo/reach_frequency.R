library(plotly)
library(ggplot2)
library(Robyn)
library(reticulate)

# simulate reach
reach = seq(from = 0, to = 10000000, by = 10000)
alpha_reach <- 1
gamma_reach <- 0.99
reach_saturated = Robyn::saturation_hill(x = reach, alpha = alpha_reach, gamma = gamma_reach)
df_reach <- data.frame(
  reach = reach,
  reach_saturated = reach_saturated
)
ggplot(df_reach) + geom_line(aes(x = reach, y = reach_saturated))

# simulate freq
freq = seq(from = 0, to = 10, by = 0.01)
alpha_freq <- 5
gamma_freq <- 0.2
freq_saturated = Robyn::saturation_hill(x = freq, alpha = alpha_freq, gamma = gamma_freq)
df_freq <- data.frame(
  freq = freq,
  freq_saturated = freq_saturated
)
ggplot(df_freq) + geom_line(aes(x = freq, y = freq_saturated))

# simulate resposne
rnf_beta <- 1000000
rnf_response = reach_saturated * freq_saturated * rnf_beta

# validate rnf_beta
# mod <- lm(rnf_response ~ I(reach_saturated * freq_saturated) - 1)
# round(mod$coefficients) == rnf_beta

response_grid <- reach_saturated %*% t(freq_saturated) * rnf_beta


# set constraints / budget
set_budget <- 100000
set_cpm <- 6
buy_imps <- set_budget/set_cpm*1000
imp_grid <- reach %*% t(freq)
filter_grid <- as.integer(imp_grid > buy_imps)
dim(filter_grid) <- dim(imp_grid)
#imp_grid[filter_grid] <- Inf
#budget_grid[budget_grid == 0] <- NA
#budget_grid <- budget_grid +1
#response_grid_below <- response_grid[budget_grid]
# response_grid2 <- response_grid
# response_grid2[filter_grid] <- NA



#htmlwidgets::saveWidget(as_widget(p_surf), "~/downloads/p_surf.html")

inflexion_reach <- c(range(reach) %*% c(1 - gamma_reach, gamma_reach))
inflexion_freq <- c(range(freq) %*% c(1 - gamma_freq, gamma_freq))


eval_list_rnf <- list(
  coefs_eval = rnf_beta,
  alphas_eval = c(alpha_reach, alpha_freq),
  inflexions_eval = c(inflexion_reach, inflexion_freq),
  total_imp = buy_imps
)



fx_objective_rnf <- function(x, alpha, inflexion, coeff) {
  # Hill transformation, x is a vector of c(reach , freq)
  x_reach <- x[[1]]
  alpha_reach <- alpha[[1]]
  inflx_reach <- inflexion[[1]]
  x_freq <- x[[2]]
  alpha_freq <- alpha[[2]]
  inflx_freq <- inflexion[[2]]
  reach_saturated <- (1 + inflx_reach**alpha_reach / x_reach**alpha_reach)**-1
  freq_saturated <- (1 + inflx_freq**alpha_freq / x_freq**alpha_freq)**-1
  xOut <- reach_saturated * freq_saturated * coeff
  return(xOut)
}

# https://www.derivative-calculator.net/ on the objective function (1/(1+gamma^alpha / x^alpha)) * (1/(1+gamma^alpha / y^alpha))
fx_gradient_rnf <- function(x, alpha, inflexion, coeff) {
  x_reach <- x[[1]]
  alpha_reach <- alpha[[1]]
  inflx_reach <- inflexion[[1]]
  x_freq <- x[[2]]
  alpha_freq <- alpha[[2]]
  inflx_freq <- inflexion[[2]]

  dv_reach <- alpha_reach * inflx_reach**alpha_reach * x_reach**(-alpha_reach-1) /
    ((inflx_freq**alpha_freq / x_freq**alpha_freq + 1) * (inflx_reach**alpha_reach / x_reach**alpha_reach + 1)**2)

  dv_freq <- alpha_freq * inflx_freq**alpha_freq * x_freq**(-alpha_freq-1) /
    ((inflx_reach**alpha_reach / x_reach**alpha_reach + 1) * (inflx_freq**alpha_freq / x_freq**alpha_freq + 1)**2)

  xOut <- c(dv_reach, dv_freq) * coeff
  return(xOut)
}

fx_objective_rnf.chanel <- function(x, alpha, inflexion, coeff) {
  x_reach <- x[[1]]
  alpha_reach <- alpha[[1]]
  inflx_reach <- inflexion[[1]]
  x_freq <- x[[2]]
  alpha_freq <- alpha[[2]]
  inflx_freq <- inflexion[[2]]

  reach_saturated <- (1 + inflx_reach**alpha_reach / x_reach**alpha_reach)**-1
  freq_saturated <- (1 + inflx_freq**alpha_freq / x_freq**alpha_freq)**-1
  xOut <- reach_saturated * freq_saturated * coeff
  return(xOut)
}


eval_f_rnf <- function(X, eval_list_rnf) {

  coefs_eval <- eval_list_rnf[["coefs_eval"]]
  alphas_eval <- eval_list_rnf[["alphas_eval"]]
  inflexions_eval <- eval_list_rnf[["inflexions_eval"]]

  objective <- - fx_objective_rnf(
    x = X,
    alpha = alphas_eval,
    inflexion = inflexions_eval,
    coeff = coefs_eval
  )

  gradient <- - fx_gradient_rnf(
    x = X,
    alpha = alphas_eval,
    inflexion = inflexions_eval,
    coeff = coefs_eval
  )

  objective.channel <- - fx_objective_rnf.chanel(
    x = X,
    alpha = alphas_eval,
    inflexion = inflexions_eval,
    coeff = coefs_eval
  )

  optm <- list(objective = objective, gradient = gradient, objective.channel = objective.channel)
  return(optm)
}


eval_g_eq_rnf <- function(X, eval_list_rnf) {

  constr <- prod(X) - eval_list_rnf$total_imp
  grad <- c(X[[2]], X[[1]])
  return(list(
    "constraints" = constr,
    "jacobian" = grad
  ))
}

eval_g_ineq_rnf <- function(X, eval_list_rnf) {

  constr <- prod(X) - eval_list_rnf$total_imp
  grad <- c(X[[2]], X[[1]])
  return(list(
    "constraints" = constr,
    "jacobian" = grad
  ))
}


x0 <- c(1, 1)
lb <- c(0, 0)
ub <- c(10000000, 10)
maxeval <- 100000
#xtol_rel <- 0.00000001

coefs_eval <- eval_list_rnf[["coefs_eval"]]
alphas_eval <- eval_list_rnf[["alphas_eval"]]
inflexions_eval <- eval_list_rnf[["inflexions_eval"]]

fx_objective_rnf(x = x0, alpha = alphas_eval, inflexion = inflexions_eval, coeff = coefs_eval)
fx_gradient_rnf(x = x0, alpha = alphas_eval, inflexion = inflexions_eval, coeff = coefs_eval)
fx_objective_rnf.chanel(x = x0, alpha = alphas_eval, inflexion = inflexions_eval, coeff = coefs_eval)


if (TRUE) {
  local_opts <- list(
    "algorithm" = "NLOPT_LD_SLSQP",
    "xtol_rel" = 1e-20,
    #"xtol_abs" = c(1000, 0.0001),
    "ftol_rel" = 1e-20,
    "maxeval" = maxeval
  )

  nlsMod <- nloptr::nloptr(
    x0 = x0,
    eval_f = eval_f_rnf,
    #eval_g_eq =  eval_g_eq_rnf,
    eval_g_ineq =  eval_g_ineq_rnf,
    lb = lb, ub = ub,
    opts = list(
      "algorithm" = "NLOPT_LD_AUGLAG",
      "xtol_rel" = 1e-20,
      #"xtol_abs" = c(100000, 0.0001),
      "ftol_rel" = 1e-20,
      "maxeval" = maxeval,
      "local_opts" = local_opts
    ),
    eval_list_rnf = eval_list_rnf
  )

  p_surf <- plot_ly(
    x = ~ freq, y = ~ reach, z = ~ response_grid,
    # contours = list(
    #   x = list(show = TRUE, start = 1.5, end = 3, size = 0.01, color = 'gray')),
    #z = list(show = TRUE, start = 0.5, end = 0.8, size = 0.05)),
  ) %>% add_surface(
    surfacecolor = filter_grid,
    colors = colorRamp(c( "steelblue", "darkgrey")),
    opacity = 0.8,
    cmin = min(filter_grid),
    cmax = max(filter_grid),
    showscale=FALSE
  ) %>% layout(scene = list(
    xaxis = list(title = 'Frequency'),
    yaxis = list(title = 'Reach'),
    zaxis = list(title = 'Sales')
  ))

  p_surf <- p_surf %>% add_trace( # for every point, add a line to the "floor"
    x = rep(nlsMod$solution[[2]],2), y = rep(nlsMod$solution[[1]],2),
    z = c(-nlsMod$objective, 0), type = 'scatter3d', mode = "lines+markers",
    color = I("red")
  ) %>% layout(
    title = paste0("Reach & frequency optimisation surface",
                   "\n(blue is budget constrained area,",
                   "\ntotal imp inventory: ",set_budget,"/",set_cpm,"*1000=",round(buy_imps),
                   "\nnonlinear optm solution: ", round(nlsMod$solution[[1]])," reach x ",
                   round(nlsMod$solution[[2]],2), " frequency)")
  )

  print(nlsMod)
  set_budget
  set_cpm
  buy_imps
  nlsMod$objective
  round(nlsMod$solution,2); prod(nlsMod$solution)
  print(paste0("validate solution vs simulated input:", round(prod(nlsMod$solution)/100) == round(buy_imps/100)))
}
p_surf
nlsMod$objective

#nloptr::nloptr.print.options()
# x1 <- 7000001
# xx <- fx_objective_rnf(x = c(8000000, 10),
#                  alpha =  eval_list_rnf[["alphas_eval"]],
#                  inflexion = eval_list_rnf[["inflexions_eval"]],
#                  coeff = eval_list_rnf[["coefs_eval"]])
# xx2 <- fx_objective_rnf(x = c(8000001, 10),
#                        alpha =  eval_list_rnf[["alphas_eval"]],
#                        inflexion = eval_list_rnf[["inflexions_eval"]],
#                        coeff = eval_list_rnf[["coefs_eval"]])
# xxx <- fx_gradient_rnf(x = c(x1, 2.663916),
#                        alpha =  eval_list_rnf[["alphas_eval"]],
#                        inflexion = eval_list_rnf[["inflexions_eval"]],
#                        coeff = eval_list_rnf[["coefs_eval"]])
# c(xx, xxx)
# sim_reach_dv <- c()
# for (i in seq_along(reach)) {
#   sim_reach_dv[i] <- fx_gradient_rnf(x = c(reach[i], 10),
#                                     alpha =  eval_list_rnf[["alphas_eval"]],
#                                     inflexion = eval_list_rnf[["inflexions_eval"]],
#                                     coeff = eval_list_rnf[["coefs_eval"]])[[1]]
# }
# ggplot(data.frame(x = reach, y = sim_reach_dv))+ geom_line(aes(x = x, y = y))
#
# sim_freq_dv <- c()
# for (i in seq_along(freq)) {
#   sim_freq_dv[i] <- fx_gradient_rnf(x = c(10000000, freq[i]),
#                              alpha =  eval_list_rnf[["alphas_eval"]],
#                              inflexion = eval_list_rnf[["inflexions_eval"]],
#                              coeff = eval_list_rnf[["coefs_eval"]])[[2]]
# }
#
# ggplot(data.frame(x = freq, y = sim_freq_dv))+ geom_line(aes(x = x, y = y))


## Decompose imp into R&F
## fit RnF to main mod saturation with nevergrad
## main model: 1st multiply rnf == imp, then lag * adstock_vec, then vec sum, then hill
r_vec1 <- c(10000, 14000, 12000, 15000)
f_vec1 <- c(2.2, 2.5, 2.3, 2.1)
imp_vec1 <- r_vec1 * f_vec1
adst_vec <- c(0.8, 1, 0.5, 0.2)

vec_collect <- list()
for (i in seq_along(imp_vec1)) {
  vec_collect[[i]] <- lag(imp_vec1[i] * adst_vec, n = i-1, default = 0)
}
vec_collect <- sapply(vec_collect, rbind)
adstocked_vec <- rowSums(vec_collect)
alpha_imp <- 1
gamma_imp <- 0.5
saturated_vec <- saturation_hill(adstocked_vec, alpha_imp, gamma_imp)

## rnf model: 1st lag  reach and lag freq individually, then vec sum, then hill individually

r_vec_collect <- list()
for (i in seq_along(r_vec1)) {
  r_vec_collect[[i]] <- lag(r_vec1[i] * adst_vec, n = i-1, default = 0)
}
r_vec_collect <- sapply(r_vec_collect, rbind)
r_adstocked_vec <- rowSums(r_vec_collect)
r_saturated_vec <- saturation_hill(r_adstocked_vec, 0.1, 0.5)

f_vec_collect <- list()
for (i in seq_along(f_vec1)) {
  f_vec_collect[[i]] <- lag(f_vec1[i] * adst_vec, n = i-1, default = 0)
}
f_vec_collect <- sapply(f_vec_collect, rbind)
f_adstocked_vec <- rowSums(f_vec_collect)
f_saturated_vec <- saturation_hill(f_adstocked_vec, 0.1, 0.5)
rnf_saturated_vec <- sqrt(r_saturated_vec * f_saturated_vec)

df_sat <- data.frame(x = rep(seq_along(saturated_vec),2) ,
                     saturated = c(saturated_vec, rnf_saturated_vec),
                     type = c(rep("imp_sat", length(saturated_vec)),
                              rep("rnf_sat", length(saturated_vec))))
df_sat %>% ggplot(aes(x = x, y = saturated, color = type)) + geom_line()

saturation_hill(1:1000, 0.1, 0.5)
# saturated_vec = sqrt(hill(r_adstocked, alpha_reach, gamma_reach) * hill(f_adstocked, alpha_freq, gamma_freq))

## rnf model: estimate alpha & gamma for reach and freq separately

if (reticulate::py_module_available("nevergrad")) {
  ng <- reticulate::import("nevergrad", delay_load = TRUE)
}

ng <- reticulate::import("nevergrad", delay_load = TRUE)
my_tuple <- reticulate::tuple(as.integer(4))
instrumentation <- ng$p$Array(shape = my_tuple, lower = 0, upper = 1)
optimizer <- ng$optimizers$registry["TwoPointsDE"](instrumentation, budget = 1000)

hp_bounds <- list(alpha_reach = c(0, 10),
                  gamma_reach = c(0,1),
                  alpha_freq = c(0, 10),
                  gamma_freq = c(0,1))
ng_hp <- list()
mse_collect <- c()
rsq_collect <- c()
pred_collect <- list()
for (i in 1:10000) {
  ng_hp[[i]] <- optimizer$ask()
  ng_hp_val <- ng_hp[[i]]$value
  ng_hp_val_scaled <- mapply(function(hpb, hp) {
    qunif(hp, min = min(hpb), max = max(hpb))
    },
    hpb = hp_bounds,
    hp = ng_hp_val)

  alpha_reach <- ng_hp_val_scaled[1]
  gamma_reach <- ng_hp_val_scaled[2]
  alpha_freq <- ng_hp_val_scaled[3]
  gamma_freq <- ng_hp_val_scaled[4]
  # inflx1 <- c(range(r_adstocked_vec) %*% c(1 - gamma_reach, gamma_reach))
  # inflx2 <- c(range(f_adstocked_vec) %*% c(1 - gamma_freq, gamma_freq))
  # saturated_vec_pred <- sqrt((r_adstocked_vec**alpha_reach / (r_adstocked_vec**alpha_reach + inflx1**alpha_reach)) *
  #   (f_adstocked_vec**alpha_freq / (f_adstocked_vec**alpha_freq + inflx2**alpha_freq)))

  ## predict imp saturation vector
  saturated_vec_pred <- sqrt(saturation_hill(r_adstocked_vec, alpha_reach, gamma_reach) *
                               saturation_hill(f_adstocked_vec, alpha_freq, gamma_freq))
  mse_iter <- mean((saturated_vec - saturated_vec_pred)**2)
  rsq_iter <- Robyn:::get_rsq(saturated_vec, saturated_vec_pred)
  pred_collect[[i]] <- saturated_vec_pred
  optimizer$tell(ng_hp[[i]], tuple(mse_iter))
  mse_collect[i] <- mse_iter
  rsq_collect[i] <- rsq_iter
}

best_iter <- which.min(mse_collect)
ng_hp[[best_iter]]$value
best_hp <- ng_hp[[best_iter]]$value
sim_r <- saturation_hill(reach, best_hp[1], best_hp[2])
sim_f <- saturation_hill(freq, best_hp[3], best_hp[4])
p1 <- data.frame(reach=reach, sat=sim_r) %>% ggplot(aes(x=reach, y=sat)) + geom_line()
p2 <- data.frame(freq=freq, sat=sim_f) %>% ggplot(aes(x=freq, y=sat)) + geom_line()
imps <- seq(0, max(adstocked_vec), 1001)
sim_imp <- saturation_hill(imps, alpha_imp, gamma_imp)
p3 <- data.frame(imps=imps, sat=sim_imp) %>% ggplot(aes(x=imps, y=sat)) + geom_line()
## imp decomposition into R&F
p3+p1+p2

p_mse <- data.frame(iterations=seq_along(mse_collect), mse = mse_collect) %>%
  ggplot(aes(x=iterations, y=mse))+geom_line()+
  labs(title = paste0("Predict saturated imp. MSE convergence with error reduction of ",
                      round((1-mse_collect[best_iter]/max(mse_collect)),8)*100, "%"))
p_rsq <- data.frame(iterations=seq_along(rsq_collect), adj.rsq = rsq_collect) %>%
  ggplot(aes(x=iterations, y=adj.rsq))+geom_line() +
  labs(title = paste0("Predict saturated imp. Adj.R2 convergence with highest value: ", round(rsq_collect[best_iter],8)))
p_mse / p_rsq


