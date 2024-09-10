library(plotly)
library(ggplot2)
library(Robyn)

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
