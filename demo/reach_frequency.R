library(plotly)
library(ggplot2)
library(Robyn)
library(reticulate)

# use Tesco reach to simulate spend
cum_reach_1plus <- c(724843, 5869128, 7497844, 13394242, 15533949, 15936122, 16431942, 16773965, 16798111)
cum_reach_1plus <- c(0, cum_reach_1plus)
cpm_range <- c(1, 100)
avg_freq_1plus <- 5.06
net_new_reach = cum_reach_1plus - lag(cum_reach_1plus, 1, 0)
cum_reach_pctl <- cum_reach_1plus / (max(cum_reach_1plus) - min(cum_reach_1plus))

.dot_product <- function(range, percentage) {
  mapply(function(percentage) {
    c(range %*% c(1 - percentage, percentage))
  },
  percentage = percentage)
}
.z_score <- function(x, mu, sd) (x - mu)/sd

.ci <- function(mu, z, sd, n) {
  c(mu - z * sd / sqrt(n), mu + z * sd / sqrt(n))
}

.qti <- function(x, qt_low = 0.025, qt_up = 0.975) c(quantile(x, qt_low), quantile(x, qt_up))
.qt_range <- function(set_range) {
  int <- (1 - set_range)/2
  return(c(low = 0 + int, up = 1 - int))
  }

geom_density_ci <- function(
    gg_density,  # ggplot object that has geom_density
    ci_low,
    ci_up,
    fill = "grey"
) {
  build_object <- ggplot_build(gg_density)
  x_dens <- build_object$data[[1]]$x
  y_dens <- build_object$data[[1]]$y
  ind_low <- min(which(x_dens >= ci_low))
  ind_up <- max(which(x_dens <= ci_up))

  gg_density <- gg_density +
    geom_area(
    data=data.frame(
      x=x_dens[ind_low:ind_up],
      density=y_dens[ind_low:ind_up]),
    aes(x=x,y=density),
    fill=fill,
    alpha=0.6)
  return(gg_density)
}

cpm_sim <- .dot_product(cpm_range, cum_reach_pctl)
cum_spend_sim = cumsum(net_new_reach * avg_freq_1plus /1000 * cpm_sim)
df_rnf_sim <- data.frame(cum_reach_1plus = cum_reach_1plus) %>%
  mutate(net_new_reach = net_new_reach,
         cum_reach_pctl = cum_reach_pctl,
         cpm_sim = cpm_sim,
         cum_spend_sim = cum_spend_sim)
df_rnf_sim %>% ggplot(aes(x = cum_spend_sim, y = cum_reach_1plus)) + geom_point()

## fit evidence
# initiate nevergrad

if (TRUE) {
  true_cum_reach <- cum_reach_1plus
  hp_bounds <- list(alpha_cf = c(0, 10), gamma_cf = c(0,1)
                    #, coef_cf = c(0, max(true_cum_reach)^2)
  )
  mse_loss <- function(y, y_hat) mean((y - y_hat)^2)

  ng_hp <- list()
  loss_collect <- c()
  rsq_collect <- c()
  pred_collect <- list()
  max_trials <- 3
  max_iters <- 2500
  loss_min_change <- 0.01
  loss_stop_ratio <- 0.05
  loss_stop_unit <- round(max_iters * loss_stop_ratio)

  if (reticulate::py_module_available("nevergrad")) {
    ng <- reticulate::import("nevergrad", delay_load = TRUE)
  }

  max_iters <- rep(max_iters, max_trials)
  for (j in 1:max_trials) {

    my_tuple <- reticulate::tuple(as.integer(2))
    instrumentation <- ng$p$Array(shape = my_tuple, lower = 0, upper = 1)
    optimizer <- ng$optimizers$registry["TwoPointsDE"](instrumentation, budget = 1000)

    ng_hp_i <- list()
    loss_collect_i <- c()
    rsq_collect_i <- c()
    pred_collect_i <- list()
    pb_cf <- txtProgressBar(min = 0, max = max_iters[j], style = 3)
    loop_continue <- TRUE
    i = 0
    while (loop_continue) {
      i <- i +1
      setTxtProgressBar(pb_cf, i)
      ng_hp_i[[i]] <- optimizer$ask()
      ng_hp_val <- ng_hp_i[[i]]$value
      ng_hp_val_scaled <- mapply(function(hpb, hp) {
        qunif(hp, min = min(hpb), max = max(hpb))
      },
      hpb = hp_bounds,
      hp = ng_hp_val)

      alpha_cf <- ng_hp_val_scaled["alpha_cf"]
      gamma_cf <- ng_hp_val_scaled["gamma_cf"]
      #coef_cf <- ng_hp_val_scaled["coef_cf"]

      ## predict imp saturation vector
      pred_cum_reach <-  saturation_hill(cum_spend_sim, alpha_cf, gamma_cf)[["x_saturated"]]
      true_cum_reach_scaled <- .min_max_norm(true_cum_reach)
      pred_cum_reach_scaled <- pred_cum_reach

      loss_iter <- mse_loss(true_cum_reach_scaled, pred_cum_reach_scaled)
      rsq_iter <- Robyn:::get_rsq(true_cum_reach, pred_cum_reach)
      pred_collect_i[[i]] <- pred_cum_reach
      optimizer$tell(ng_hp_i[[i]], tuple(loss_iter))
      loss_collect_i[i] <- loss_iter
      rsq_collect_i[i] <- rsq_iter

      max_loss <- ifelse(i==1, loss_iter, max(max_loss, loss_iter))
      loss_min_change_abs <- max_loss * loss_min_change

      if ((i >= (loss_stop_unit * 2))) {
        if ((i == max_iters[j])) {
          loop_continue <- FALSE
          close(pb_cf)
          message(paste0("Trial ", j, " didn't converged after ", i, " iterations. Increase iterations or adjust convergence criterias."))
        } else {
          current_unit <- (i-loss_stop_unit+1):i
          previous_unit <- current_unit-loss_stop_unit
          loss_unit_change <- (mean(loss_collect_i[current_unit]) - mean(loss_collect_i[previous_unit]))
          loop_continue <- !all(loss_unit_change > 0, loss_unit_change <= loss_min_change_abs)

          if (loop_continue == FALSE) {
            close(pb_cf)
            message(paste0("Trial ", j, " converged & stopped at iteration ", i, " from ", max_iters[j]))
            max_iters[j] <- i
          }
        }
      }
    }
    ng_hp[[j]] <- ng_hp_i
    loss_collect[[j]] <- loss_collect_i
    rsq_collect[[j]] <- rsq_collect_i
    pred_collect[[j]] <- pred_collect_i
    close(pb_cf)
  }

  best_loss_iters <- mapply(function(x) which.min(x), x = loss_collect)
  best_loss_vals <- mapply(function(x) min(x), x = loss_collect)
  best_loss_trial <- which.min(best_loss_vals)
  best_loss_iter <- best_loss_iters[best_loss_trial]
  best_loss_val <- best_loss_vals[best_loss_trial]
  best_hp <- ng_hp[[best_loss_trial]][[best_loss_iter]]$value

  best_alpha <- .dot_product(hp_bounds[[1]], best_hp[1])
  best_gamma <- .dot_product(hp_bounds[[2]], best_hp[2])
  best_pred_cum_reach <- saturation_hill(cum_spend_sim, best_alpha, best_gamma)[["x_saturated"]]
  best_inflexion <- saturation_hill(cum_spend_sim, best_alpha, best_gamma)[["inflexion"]]


  alpha_collect <- lapply(ng_hp, FUN = function(x) {sapply(x, FUN = function(y) .dot_product(hp_bounds[[1]], y$value[1]))})
  gamma_collect <- lapply(ng_hp, FUN = function(x) {sapply(x, FUN = function(y) .dot_product(hp_bounds[[2]], y$value[2]))})
  burn_in_iters <- rep(loss_stop_unit * 2, length(max_iters))

  alpha_collect_converged <- unlist(mapply(function(x, y, z) x[y:z],
                                           x = alpha_collect, y = burn_in_iters,
                                           z = max_iters, SIMPLIFY = FALSE))
  gamma_collect_converged <- unlist(mapply(function(x, y, z) x[y:z],
                                           x = gamma_collect, y = burn_in_iters,
                                           z = max_iters, SIMPLIFY = FALSE))

  s_alpha <- sd(alpha_collect_converged)
  mu_alpha <- mean(alpha_collect_converged)
  md_alpha <- median(alpha_collect_converged)
  n_alpha <- length(alpha_collect_converged)
  z_alpha <- .z_score(x = alpha_collect_converged, mu = mu_alpha, sd = s_alpha)

  s_gamma <- sd(gamma_collect_converged)
  mu_gamma <- mean(gamma_collect_converged)
  md_gamma <- median(gamma_collect_converged)
  n_gamma <- length(gamma_collect_converged)
  z_gamma <- .z_score(x = gamma_collect_converged, mu = mu_gamma, sd = s_gamma)

  b <- .qt_range(0.95)
  qt_95_alpha <- .qti(x = alpha_collect_converged, qt_low = b["low"], qt_up = b["up"])
  qt_95_gamma <- .qti(x = gamma_collect_converged, qt_low = b["low"], qt_up = b["up"])





  # p_lines <- data.frame(spend = rep(cum_spend_sim, 2),
  #                       reach = c(true_cum_reach_scaled, best_pred_cum_reach),
  #                       type = c(rep("true", length(true_cum_reach)),
  #                                rep("predicted", length(true_cum_reach)))) %>%
  #   ggplot(aes(x=spend, y=reach, color = type)) + geom_line() +
  #   labs(title = "Spend to reach cumulative curve prediction")

  sim_n <- 50
  df_true <- data.frame(spend = cum_spend_sim, reach = true_cum_reach_scaled)
  temp_spend <- seq(0, max(cum_spend_sim), by = sim_n)
  temp_sat <- saturation_hill(temp_spend, best_alpha, best_gamma)[["x_saturated"]]
  df_pred <- data.frame(spend = temp_spend, reach = temp_sat)

  sim_alphas <- alpha_collect_converged[
    alpha_collect_converged > qt_95_alpha[1] &
      alpha_collect_converged < qt_95_alpha[2]]
  sim_alphas <- sample(sim_alphas, sim_n, replace = TRUE)
  sim_gammas <- gamma_collect_converged[
    gamma_collect_converged > qt_95_gamma[1] &
      gamma_collect_converged < qt_95_gamma[2]]
  sim_gammas <- sample(sim_gammas, sim_n, replace = TRUE)

  sim_collect <- list()
  for (i in 1:sim_n) {
    sim_collect[[i]] <- saturation_hill(temp_spend, sim_alphas[i], sim_gammas[i])[["x_saturated"]]
  }
  sim_collect <- data.frame(
    sim = as.character(c(sapply(1:sim_n, function(x) rep(x, length(temp_spend))))),
    sim_spend = rep(temp_spend, sim_n),
    sim_saturation = unlist(sim_collect))

  p_lines <- ggplot() +
    geom_line(data = sim_collect,
              aes(x = .data$sim_spend, y = .data$sim_saturation,
                  color = sim), size = 2, alpha = 0.2) +
    scale_colour_grey() +
    theme(legend.position="none") +
    geom_point(
      data = df_true,
      aes(x=.data$spend, y=.data$reach)) +
    geom_line(
      data = df_pred,
      aes(x=.data$spend, y=.data$reach), color = "blue") +
    labs(title = "Halo cumulative reach curve fitting")

  df_mse <- data.frame(mse = unlist(loss_collect),
             iterations = unlist(mapply(function(x) 1:x, max_iters, SIMPLIFY = FALSE)),
             trials = as.character(unlist(
               mapply(function (x, y) rep(x, y),
                      x = 1:max_trials, y = max_iters))))
  p_mse <- df_mse %>%
    ggplot(aes(x=.data$iterations, y=.data$mse))+geom_line()+facet_grid(.data$trials ~ .)+
    labs(title = paste0("Loss convergence with error reduction of ",
                        round((1-best_loss_val/max_loss),4)*100, "%"))
  # p_rsq <- data.frame(iterations=seq_along(rsq_collect), adj.rsq = rsq_collect) %>%
  #   ggplot(aes(x=iterations, y=adj.rsq))+geom_line() +
  #   labs(title = paste0("Predict saturated imp. Adj.R2 convergence with highest value: ", round(rsq_collect[best_iter],8)))




  p_alpha <- data.frame(alpha = alpha_collect_converged) %>% ggplot(aes(x = alpha)) +
    geom_density(fill = "grey99", color = "grey") +
    labs(title = "Alpha (Hill) density after 10% burn-in.",
         subtitle = paste0("95% interval: ", round(qt_95_alpha[1],4), "-", round(qt_95_alpha[2],4)))
  p_alpha <- geom_density_ci(p_alpha, qt_95_alpha[1], qt_95_alpha[2], fill = "lightblue")
  p_gamma <- data.frame(gamma = gamma_collect_converged) %>% ggplot(aes(x = gamma)) +
    geom_density(fill = "grey99", color = "grey") +
    labs(title = "Gamma (Hill) density after 10% burn-in.",
         subtitle = paste0("95% interval: ", round(qt_95_gamma[1],4), "-", round(qt_95_gamma[2],4)))
  p_gamma <- geom_density_ci(p_gamma, qt_95_gamma[1], qt_95_gamma[2], fill = "lightblue")


  print(p_lines / p_mse / (p_alpha + p_gamma))
  print(best_loss_val)
  message(paste0("best alpha: ", best_alpha, ", best gamma: ", best_gamma, " , best inflexion: ", best_inflexion))
}



ng_hp_collect



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
# rnf_saturated_vec <- sqrt(r_saturated_vec * f_saturated_vec)
#
# df_sat <- data.frame(x = rep(seq_along(saturated_vec),2) ,
#                      saturated = c(saturated_vec, rnf_saturated_vec),
#                      type = c(rep("imp_sat", length(saturated_vec)),
#                               rep("rnf_sat", length(saturated_vec))))
# df_sat %>% ggplot(aes(x = x, y = saturated, color = type)) + geom_line()

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
loss_collect <- c()
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
  loss_iter <- mean((saturated_vec - saturated_vec_pred)**2)
  rsq_iter <- Robyn:::get_rsq(saturated_vec, saturated_vec_pred)
  pred_collect[[i]] <- saturated_vec_pred
  optimizer$tell(ng_hp[[i]], tuple(loss_iter))
  loss_collect[i] <- loss_iter
  rsq_collect[i] <- rsq_iter
}

best_iter <- which.min(loss_collect)
ng_hp[[best_iter]]$value
best_hp <- ng_hp[[best_iter]]$value
sim_r <- saturation_hill(reach, best_hp[1], best_hp[2])
sim_f <- saturation_hill(freq, best_hp[3], best_hp[4])

p1 <- data.frame(reach=reach, sat=sim_r) %>% ggplot(aes(x=reach, y=sat)) + geom_line()
p2 <- data.frame(freq=freq, sat=sim_f) %>% ggplot(aes(x=freq, y=sat)) + geom_line()
imps_sim <- seq(0, max(adstocked_vec), 1001)
imps_sim_sat <- saturation_hill(imps_sim, alpha_imp, gamma_imp)
p3 <- data.frame(imps=imps_sim, sat=imps_sim_sat) %>% ggplot(aes(x=imps, y=sat)) + geom_line()
## imp decomposition into R&F
p3+p1+p2

data.frame(main_model_sat_sim = saturated_vec,
           rnf_pred_sat_sim = sqrt(sim_r * sim_f)) %>%
  ggplot(aes(x = main_model_sat_sim, y = rnf_pred_sat_sim)) + geom_line()

p_mse <- data.frame(iterations=seq_along(loss_collect), mse = loss_collect) %>%
  ggplot(aes(x=iterations, y=mse))+geom_line()+
  labs(title = paste0("Predict saturated imp. MSE convergence with error reduction of ",
                      round((1-loss_collect[best_iter]/max(loss_collect)),8)*100, "%"))
p_rsq <- data.frame(iterations=seq_along(rsq_collect), adj.rsq = rsq_collect) %>%
  ggplot(aes(x=iterations, y=adj.rsq))+geom_line() +
  labs(title = paste0("Predict saturated imp. Adj.R2 convergence with highest value: ", round(rsq_collect[best_iter],8)))
p_mse / p_rsq


