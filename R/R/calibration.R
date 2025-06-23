# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn Calibration Function - BETA
#'
#' \code{robyn_calibrate()} consumes source of truth or proxy data for
#' saturation or adstock curve estimation. This is an experimental feature and
#' can be used independently from Robyn's main model.
#'
#' @inheritParams robyn_run
#' @param df_curve data.frame. Requires two columns named spend and response.
#' Recommended sources of truth are Halo R&F or Meta conversion lift.
#' @param curve_type Character. Currently only allows "saturation_reach_hill"
#' and only supports Hill function.
#' @param force_shape Character. Allows c("c", "s") with default NULL that's no
#' shape forcing. It's recommended for offline media to have "c" shape, while
#' for online can be "s" or NULL. Shape forcing only works if hp_bounds is null.
#' @param hp_bounds list. Currently only allows Hill for saturation. Ranges
#' for alpha and gamma are provided as Hill parameters. If NULL, hp_bounds takes
#' on default ranges.
#' @param max_trials integer. Different trials have different starting point
#' and provide diversified sampling paths. Default to 10.
#' @param max_iters integer. Loss is minimized while iteration increases.
#' Default to 2500.
#' @param loss_min_step_rel numeric. Default to 0.01 and value is between 0-0.1.
#' 0.01 means the optimisation is considered converged if error minimization is
#' <1 percent of maximal error.
#' @param loss_stop_rel numeric. Default is 0.05 and value is between 0-0.5.
#' 0.05 means 5 percent of the max_iters is used as the length of iterations to
#' calculate the mean error for convergence.
#' @param burn_in_rel numeric. Default to 0.1 and value is between 0.0.5. 0.1
#' means 10 percent of iterations is used as burn-in period.
#' @param sim_n integer. Number of simulation for plotting fitted curve.
#' @param hp_interval numeric. Default to 0.95 and is between 0.8-1. 0.95 means
#' 2.5 - 97.5 percent percentile are used as parameter range for output.
#' @examples
#' \dontrun{
#' # Dummy input data for Meta spend. This is derived from Halo's reach & frequency data.
#' # Note that spend and response need to be cumulative metrics.
#' data("df_curve_reach_freq")
#'
#' # Using reach saturation from Halo as proxy
#' curve_out <- robyn_calibrate(
#'   df_curve = df_curve_reach_freq,
#'   curve_type = "saturation_reach_hill"
#' )
#' # For the simulated reach and frequency dataset, it's recommended to use
#' # "reach 1+" for gamma lower bound and "reach 10+" for gamma upper bound
#' facebook_I_gammas <- c(
#'   curve_out[["curve_collect"]][["reach 1+"]][["hill"]][["gamma_best"]],
#'   curve_out[["curve_collect"]][["reach 10+"]][["hill"]][["gamma_best"]]
#' )
#' print(facebook_I_gammas)
#' }
#' @return List. Class: \code{curve_out}. Contains the results of all trials
#' and iterations modeled.
#' @export
robyn_calibrate <- function(
    df_curve = NULL,
    curve_type = NULL,
    force_shape = NULL,
    hp_bounds = NULL,
    max_trials = 10,
    max_iters = 2500,
    loss_min_step_rel = 0.0001,
    loss_stop_rel = 0.05,
    burn_in_rel = 0.1,
    sim_n = 30,
    hp_interval = 0.5,
    quiet = FALSE,
    ...) {
  ## check all inputs
  # df_curve df format
  # curve types
  # hp_bounds format
  # hp_interval

  if (curve_type == "saturation_reach_hill") {
    curve_collect <- list()
    for (i in unique(df_curve$freq_bucket)) {
      message(">>> Fitting ", i)
      df_curve_i <- df_curve %>% filter(.data$freq_bucket == i)
      curve_collect[[i]] <- robyn_calibrate_single_dim(
        df_curve = df_curve_i,
        curve_type,
        force_shape = "c", # assumption: reach saturation is always concave
        hp_bounds,
        max_trials,
        max_iters,
        loss_min_step_rel,
        loss_stop_rel,
        burn_in_rel,
        sim_n,
        hp_interval,
        quiet
      )
    }

    df_curve_plot <- bind_rows(lapply(curve_collect, function(x) x$df_out))

    p_rnf <- ggplot(df_curve_plot, aes(x = .data$spend_cumulated)) +
      geom_point(aes(y = .data$response_cumulated, colour = .data$freq_bucket)) +
      geom_line(aes(y = .data$response_pred, colour = .data$freq_bucket), alpha = 0.5) +
      labs(
        title = "Cumulative reach & frequency saturation fitting",
        subtitle = "The dots are input R&F data. The lines are fitted curves.",
        x = "cumulative spend",
        y = "cumulative reach"
      ) +
      # theme_lares(background = "white")+
      # scale_alpha_discrete(range = c(1, 0.2))
      scale_colour_discrete(h = c(120, 260))

    return(list(
      curve_collect = curve_collect,
      plot_reach_freq = p_rnf
    ))
  } else {
    curve_collect <- robyn_calibrate_single_dim(
      df_curve,
      curve_type,
      force_shape,
      hp_bounds,
      max_trials,
      max_iters,
      loss_min_step_rel,
      loss_stop_rel,
      burn_in_rel,
      sim_n,
      hp_interval,
      quiet
    )
    return(list(curve_collect = curve_collect))
  }
}


robyn_calibrate_single_dim <- function(
    df_curve,
    curve_type,
    force_shape,
    hp_bounds,
    max_trials,
    max_iters,
    loss_min_step_rel,
    loss_stop_rel,
    burn_in_rel,
    sim_n,
    hp_interval,
    quiet,
    ...) {
  spend_cum_sot <- df_curve[["spend_cumulated"]]
  response_cum_sot <- df_curve[["response_cumulated"]]
  # amend 0 if not available
  if (!any(spend_cum_sot == 0)) {
    spend_cum_sot <- c(0, spend_cum_sot)
    response_cum_sot <- c(0, response_cum_sot)
  }

  ## get hyperparameter bounds
  if (is.null(hp_bounds)) {
    hp_bounds <- list(hill = list(alpha = c(0, 10), gamma = c(0, 1)), coef = c(0, max(response_cum_sot) / 0.01))
    hp_bounds_loop <- hp_bounds[["hill"]]
    hp_bounds_loop[["coef"]] <- hp_bounds[["coef"]]
    if (force_shape == "s") {
      hp_bounds_loop[["alpha"]] <- c(1, 10)
    } else if (force_shape == "c") {
      hp_bounds_loop[["alpha"]] <- c(0, 1)
    }
  } else {
    hp_bounds_loop <- hp_bounds[["hill"]]
    hp_bounds_loop[["coef"]] <- hp_bounds[["coef"]]
  }

  ## initiate Nevergrad
  if (reticulate::py_module_available("nevergrad")) {
    ng <- reticulate::import("nevergrad", delay_load = TRUE)
  }

  ## trial loop
  ng_hp <- list()
  loss_collect <- c()
  pred_collect <- list()
  loss_stop_abs <- round(max_iters * loss_stop_rel)
  max_iters_vec <- rep(max_iters, max_trials)

  for (j in seq(max_trials)) {
    my_tuple <- reticulate::tuple(as.integer(3))
    instrumentation <- ng$p$Array(shape = my_tuple, lower = 0, upper = 1)
    optimizer <- ng$optimizers$registry["TwoPointsDE"](instrumentation, budget = max_iters)

    ## inner while loop that stops when converged
    ng_hp_i <- list()
    loss_collect_i <- c()
    pred_collect_i <- list()
    if (!quiet) pb_cf <- txtProgressBar(min = 0, max = max_iters_vec[j], style = 3)
    loop_continue <- TRUE
    i <- 0

    while (loop_continue) {
      i <- i + 1
      if (!quiet) setTxtProgressBar(pb_cf, i)

      ## Nevergrad ask sample
      ng_hp_i[[i]] <- optimizer$ask()
      ng_hp_val <- ng_hp_i[[i]]$value
      ng_hp_val_scaled <- mapply(
        function(hpb, hp) {
          qunif(hp, min = min(hpb), max = max(hpb))
        },
        hpb = hp_bounds_loop,
        hp = ng_hp_val
      )
      alpha <- ng_hp_val_scaled["alpha"]
      gamma <- ng_hp_val_scaled["gamma"]
      coeff <- ng_hp_val_scaled["coef"]

      ## predict saturation vector
      total_cum_spend <- max(spend_cum_sot)
      response_pred <- coeff * saturation_hill(x = total_cum_spend, alpha, gamma, x_marginal = spend_cum_sot)[["x_saturated"]]
      # response_sot_scaled <- .min_max_norm(response_cum_sot)

      ## get loss
      loss_iter <- sqrt(.mse_loss(y = response_cum_sot, y_hat = response_pred))
      max_loss <- ifelse(i == 1, loss_iter, max(max_loss, loss_iter))
      loss_min_step_abs <- max_loss * loss_min_step_rel

      ## collect loop results
      pred_collect_i[[i]] <- response_pred
      loss_collect_i[i] <- loss_iter

      ## Nevergrad tell loss
      optimizer$tell(ng_hp_i[[i]], tuple(loss_iter))

      ## Loop config & prompting
      if ((i >= (loss_stop_abs * 2))) {
        if ((i == max_iters_vec[j])) {
          loop_continue <- FALSE
          if (!quiet) {
            close(pb_cf)
            message(paste0(
              "Trial ", j, " didn't converged after ", i,
              " iterations. Increase iterations or adjust convergence criterias."
            ))
          }
        } else {
          current_unit <- (i - loss_stop_abs + 1):i
          previous_unit <- current_unit - loss_stop_abs
          loss_unit_change <- (mean(loss_collect_i[current_unit]) - mean(loss_collect_i[previous_unit]))
          loop_continue <- !all(loss_unit_change > 0, loss_unit_change <= loss_min_step_abs)

          if (loop_continue == FALSE) {
            if (!quiet) {
              close(pb_cf)
              message(paste0(
                "Trial ", j, " converged & stopped at iteration ", i,
                " from ", max_iters_vec[j]
              ))
            }
            max_iters_vec[j] <- i
          }
        }
      }
    }
    ng_hp[[j]] <- ng_hp_i
    loss_collect[[j]] <- loss_collect_i
    pred_collect[[j]] <- pred_collect_i
    if (!quiet) close(pb_cf)
  }

  ## collect loop output
  best_loss_iters <- mapply(function(x) which.min(x), x = loss_collect)
  best_loss_vals <- mapply(function(x) min(x), x = loss_collect)
  best_loss_trial <- which.min(best_loss_vals)
  best_loss_iter <- best_loss_iters[best_loss_trial]
  best_loss_val <- best_loss_vals[best_loss_trial]
  best_hp <- ng_hp[[best_loss_trial]][[best_loss_iter]]$value
  best_pred_response <- pred_collect[[best_loss_trial]][[best_loss_iter]]

  ## saturation hill
  hp_alpha <- hp_bounds_loop[["alpha"]]
  hp_gamma <- hp_bounds_loop[["gamma"]]
  hp_coef <- hp_bounds_loop[["coef"]]
  best_alpha <- qunif(best_hp[1], min = min(hp_alpha), max = max(hp_alpha))
  best_gamma <- qunif(best_hp[2], min = min(hp_gamma), max = max(hp_gamma))
  best_coef <- qunif(best_hp[3], min = min(hp_coef), max = max(hp_coef))
  # best_response_pred <- saturation_hill(spend_cum_sot, best_alpha, best_gamma)[["x_saturated"]]
  # best_inflexion <- saturation_hill(spend_cum_sot, best_alpha, best_gamma)[["inflexion"]]
  alpha_collect <- lapply(ng_hp, FUN = function(x) {
    sapply(x, FUN = function(y) qunif(y$value[1], min = min(hp_alpha), max = max(hp_alpha)))
  })
  gamma_collect <- lapply(ng_hp, FUN = function(x) {
    sapply(x, FUN = function(y) qunif(y$value[2], min = min(hp_gamma), max = max(hp_gamma)))
  })
  coef_collect <- lapply(ng_hp, FUN = function(x) {
    sapply(x, FUN = function(y) qunif(y$value[3], min = min(hp_coef), max = max(hp_coef)))
  })

  ## slice by convergence
  burn_in_abs <- rep(max_iters * burn_in_rel, max_trials)
  alpha_collect_converged <- unlist(mapply(
    function(x, start, end) x[start:end],
    x = alpha_collect, start = burn_in_abs,
    end = max_iters_vec, SIMPLIFY = FALSE
  ))
  gamma_collect_converged <- unlist(mapply(
    function(x, start, end) x[start:end],
    x = gamma_collect, start = burn_in_abs,
    end = max_iters_vec, SIMPLIFY = FALSE
  ))
  coef_collect_converged <- unlist(mapply(
    function(x, start, end) x[start:end],
    x = coef_collect, start = burn_in_abs,
    end = max_iters_vec, SIMPLIFY = FALSE
  ))

  ## get calibration range for hyparameters
  p_alpha <- data.frame(alpha = alpha_collect_converged) %>%
    ggplot(aes(x = alpha)) +
    geom_density(fill = "grey99", color = "grey")
  alpha_den <- .den_interval(p_alpha, hp_interval, best_alpha)

  p_gamma <- data.frame(gamma = gamma_collect_converged) %>% ggplot(aes(x = gamma)) +
    geom_density(fill = "grey99", color = "grey")
  gamma_den <- .den_interval(p_gamma, hp_interval, best_gamma)

  p_coef <- data.frame(coef = coef_collect_converged) %>% ggplot(aes(x = coef)) +
    geom_density(fill = "grey99", color = "grey")
  coef_den <- .den_interval(p_coef, hp_interval, best_coef)

  # qt_alpha_out <- .qti(x = alpha_collect_converged, interval = hp_interval)
  # qt_gamma_out <- .qti(x = gamma_collect_converged, interval = hp_interval)
  # qt_coef_out <- .qti(x = coef_collect_converged, interval = hp_interval)

  ## plotting & prompting
  # coef_response <- max(response_cum_sot) / max(response_sot_scaled)
  df_sot_plot <- data.frame(
    spend = spend_cum_sot,
    response = response_cum_sot,
    response_pred = best_pred_response
  )
  temp_spend <- seq(0, max(spend_cum_sot), length.out = sim_n)
  temp_sat <- best_coef * saturation_hill(x = total_cum_spend, alpha = best_alpha, gamma = best_gamma, x_marginal = temp_spend)[["x_saturated"]]
  df_pred_sim_plot <- data.frame(spend = temp_spend, response = temp_sat)

  sim_alphas <- alpha_collect_converged[
    alpha_collect_converged > alpha_den$interval[1] &
      alpha_collect_converged < alpha_den$interval[2]
  ]
  sim_alphas <- sample(sim_alphas, sim_n, replace = TRUE)
  sim_gammas <- gamma_collect_converged[
    gamma_collect_converged > gamma_den$interval[1] &
      gamma_collect_converged < gamma_den$interval[2]
  ]
  sim_gammas <- sample(sim_gammas, sim_n, replace = TRUE)

  # simulation for plotting
  sim_collect <- list()
  for (i in 1:sim_n) {
    sim_collect[[i]] <- best_coef * saturation_hill(x = total_cum_spend, alpha = sim_alphas[i], gamma = sim_gammas[i], x_marginal = temp_spend)[["x_saturated"]]
  }
  sim_collect <- data.frame(
    sim = as.character(c(sapply(1:sim_n, function(x) rep(x, length(temp_spend))))),
    sim_spend = rep(temp_spend, sim_n),
    sim_saturation = unlist(sim_collect)
  )

  y_lab <- "response proxy"
  p_lines <- ggplot() +
    geom_line(
      data = sim_collect,
      aes(
        x = .data$sim_spend, y = .data$sim_saturation,
        color = .data$sim
      ), linewidth = 2, alpha = 0.2
    ) +
    scale_colour_grey() +
    geom_point(
      data = df_sot_plot,
      aes(x = .data$spend, y = .data$response)
    ) +
    geom_line(
      data = df_pred_sim_plot,
      aes(x = .data$spend, y = .data$response), color = "blue"
    ) +
    labs(title = paste0("Spend to ", y_lab, " saturation curve estimation")) +
    ylab(y_lab) +
    xlab("Spend") +
    theme_lares(legend = "none", ...)

  df_mse <- data.frame(
    mse = unlist(loss_collect),
    iterations = unlist(mapply(function(x) seq(x), max_iters_vec, SIMPLIFY = FALSE)),
    trials = as.character(unlist(
      mapply(function(x, y) rep(x, y),
        x = 1:max_trials, y = max_iters_vec
      )
    ))
  )
  p_mse <- df_mse %>%
    mutate(trials = factor(.data$trials, levels = seq(max_trials))) %>%
    ggplot(aes(x = .data$iterations, y = .data$mse)) +
    geom_line(linewidth = 0.2) +
    facet_grid(.data$trials ~ .) +
    labs(
      title = paste0(
        "Loss convergence with error reduction of ",
        round((1 - best_loss_val / max_loss), 4) * 100, "%"
      ),
      x = "Iterations", y = "MSE"
    ) +
    theme_lares(grid = "Xx", ...) +
    scale_x_abbr() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    )

  p_alpha <- p_alpha +
    labs(
      title = paste0("Alpha (Hill) density after ", round(burn_in_rel * 100), "% burn-in"),
      subtitle = paste0(
        round(hp_interval * 100), "% center density: ", round(alpha_den$interval[1], 4), "-", round(alpha_den$interval[2], 4),
        "\nBest alpha: ", round(best_alpha, 4)
      )
    ) +
    theme_lares(...) +
    scale_y_abbr()
  p_alpha <- geom_density_ci(p_alpha, alpha_den$interval[1], alpha_den$interval[2], fill = "lightblue")

  p_gamma <- p_gamma +
    labs(
      title = paste0("Gamma (Hill) density after ", round(burn_in_rel * 100), "% burn-in"),
      subtitle = paste0(
        round(hp_interval * 100), "% center density: ", round(gamma_den$interval[1], 4), "-", round(gamma_den$interval[2], 4),
        "\nBest gamma: ", round(best_gamma, 4)
      )
    ) +
    theme_lares(...) +
    scale_y_abbr()
  p_gamma <- geom_density_ci(p_gamma, gamma_den$interval[1], gamma_den$interval[2], fill = "lightblue")

  # p_coef <- p_coef +
  #   labs(
  #     title = paste0("Beta coefficient density after ", round(burn_in_rel * 100), "% burn-in"),
  #     subtitle = paste0(round(hp_interval*100), "% center density: ", round(exp(coef_den$interval[1])), "-", round(exp(coef_den$interval[2]))),
  #     x = "log(coef)"
  #   ) +
  #   theme_lares(...) +
  #   scale_y_abbr()
  # p_coef <- geom_density_ci(p_coef, coef_den$interval[1], coef_den$interval[2], fill = "lightblue")

  if (!quiet) {
    message(
      paste0(
        "\nBest alpha: ", round(best_alpha, 4), " (",
        paste0(round(alpha_den$interval, 4), collapse = "-"), ")",
        ", Best gamma: ", round(best_gamma, 4), " (",
        paste0(round(gamma_den$interval, 4), collapse = "-"), ")",
        ", Best coef: ", round(best_coef), " (",
        paste0(round(coef_den$interval), collapse = "-"), ")",
        ", Total spend: ", max(spend_cum_sot), ", Best loss: ",
        round(best_loss_val, 4), "\n"
      )
    )
  }

  curve_out <- list(
    hill = list(
      alpha_range = c(alpha_den$interval),
      alpha_best = best_alpha,
      gamma_range = c(gamma_den$interval),
      gamma_best = best_gamma,
      coef_range = c(coef_den$interval),
      coef_best = best_coef,
      inflexion_max = total_cum_spend
    ),
    plot = p_lines / p_mse / (p_alpha + p_gamma) +
      plot_annotation(
        theme = theme_lares(background = "white", ...)
      ),
    df_out = df_curve %>%
      mutate(response_pred = best_pred_response),
    df_out_sim = df_pred_sim_plot %>%
      mutate(response_pred = .data$response)
  )
  return(curve_out)
}


.den_interval <- function(plot_object, hp_interval, best_val) {
  get_den <- ggplot_build(plot_object)$data[[1]]
  # mode_loc <- which.max(get_den$y)
  mode_loc <- which.min(abs(get_den$x - best_val))
  mode_wing <- sum(get_den$y) * hp_interval / 2
  int_left <- mode_loc - which.min(abs(cumsum(get_den$y[mode_loc:1]) - mode_wing)) + 1
  int_left <- ifelse(is.na(int_left) | int_left < 1, 1, int_left)
  int_right <- mode_loc + which.min(abs(cumsum(get_den$y[(mode_loc + 1):length(get_den$y)]) - mode_wing))
  int_right <- ifelse(length(int_right) == 0, length(get_den$y), int_right)
  return(list(
    interval = c(get_den$x[int_left], get_den$x[int_right]),
    mode = get_den$x[mode_loc]
  ))
}


lift_calibration <- function(
    calibration_input,
    df_raw, # df_raw = InputCollect$dt_mod
    dayInterval, # dayInterval = InputCollect$dayInterval
    xDecompVec, # xDecompVec = decompCollect$xDecompVec
    coefs, # coefs = decompCollect$coefsOutCat
    hypParamSam,
    wind_start = 1,
    wind_end = nrow(df_raw),
    adstock) {
  ds_wind <- df_raw$ds[wind_start:wind_end]
  include_study <- any(
    calibration_input$liftStartDate >= min(ds_wind) &
      calibration_input$liftEndDate <= (max(ds_wind) + dayInterval - 1)
  )
  if (!is.null(calibration_input) & !include_study) {
    warning("All calibration_input in outside modelling window. Running without calibration")
  } else if (!is.null(calibration_input) & include_study) {
    calibration_input <- mutate(
      calibration_input,
      pred = NA, pred_total = NA, decompStart = NA, decompEnd = NA
    )
    split_channels <- strsplit(calibration_input$channel_selected, split = "\\+")

    for (l_study in seq_along(split_channels)) {
      get_channels <- split_channels[[l_study]]
      scope <- calibration_input$calibration_scope[[l_study]]
      study_start <- calibration_input$liftStartDate[[l_study]]
      study_end <- calibration_input$liftEndDate[[l_study]]
      study_pos <- which(df_raw$ds >= study_start & df_raw$ds <= study_end)
      if (study_start %in% df_raw$ds) {
        calib_pos <- study_pos
      } else {
        calib_pos <- c(min(study_pos) - 1, study_pos)
      }
      calibrate_dates <- df_raw[calib_pos, "ds"][[1]]
      calib_pos_rw <- which(xDecompVec$ds %in% calibrate_dates)

      l_chn_collect <- list()
      l_chn_total_collect <- list()
      for (l_chn in seq_along(get_channels)) { # l_chn =1
        if (scope == "immediate") {
          m <- df_raw[, get_channels[l_chn]][[1]]
          # m_calib <- df_raw[calib_pos, get_channels[l_chn]][[1]]

          ## 1. Adstock
          if (adstock == "geometric") {
            theta <- hypParamSam[paste0(get_channels[l_chn], "_thetas")][[1]][[1]]
          }
          if (grepl("weibull", adstock)) {
            shape <- hypParamSam[paste0(get_channels[l_chn], "_shapes")][[1]][[1]]
            scale <- hypParamSam[paste0(get_channels[l_chn], "_scales")][[1]][[1]]
          }
          x_list <- transform_adstock(m, adstock, theta = theta, shape = shape, scale = scale)
          x_list_cal <- transform_adstock(m[calib_pos], adstock, theta = theta, shape = shape, scale = scale)
          if (adstock == "weibull_pdf") {
            m_imme <- x_list$x_imme
            m_imme_cal <- x_list_cal$x_imme
          } else {
            m_imme <- m
            m_imme_cal <- m[calib_pos]
          }
          m_total <- x_list$x_decayed
          # m_caov <- m_total - m_imme
          m_total_cal <- x_list_cal$x_decayed

          ## 2. Saturation
          m_total_rw <- m_total[wind_start:wind_end]
          alpha <- hypParamSam[paste0(get_channels[l_chn], "_alphas")][[1]][[1]]
          gamma <- hypParamSam[paste0(get_channels[l_chn], "_gammas")][[1]][[1]]
          m_calib_caov_sat <- saturation_hill(
            m_total_rw,
            alpha = alpha, gamma = gamma, x_marginal = m_total[calib_pos] - m_total_cal
          )
          m_calib_caov_decomp <- m_calib_caov_sat$x_saturated * coefs$s0[coefs$rn == get_channels[l_chn]]
          m_calib_total_decomp <- xDecompVec[calib_pos_rw, get_channels[l_chn]]
          m_calib_decomp <- m_calib_total_decomp - m_calib_caov_decomp
        }
        if (scope == "total") {
          m_calib_decomp <- m_calib_total_decomp <- xDecompVec[calib_pos_rw, get_channels[l_chn]]
        }
        l_chn_collect[[get_channels[l_chn]]] <- m_calib_decomp
        l_chn_total_collect[[get_channels[l_chn]]] <- m_calib_total_decomp
      }

      if (length(get_channels) > 1) {
        l_chn_collect <- rowSums(bind_cols(l_chn_collect))
        l_chn_total_collect <- rowSums(bind_cols(l_chn_total_collect))
      } else {
        l_chn_collect <- unlist(l_chn_collect, use.names = FALSE)
        l_chn_total_collect <- unlist(l_chn_total_collect, use.names = FALSE)
      }

      calibration_input[l_study, ] <- mutate(
        calibration_input[l_study, ],
        pred = sum(l_chn_collect),
        pred_total = sum(l_chn_total_collect),
        decompStart = range(calibrate_dates)[1],
        decompEnd = range(calibrate_dates)[2]
      )
    }
    liftCollect <- calibration_input %>%
      mutate(
        decompStart = as.Date(.data$decompStart, "1970-01-01"),
        decompEnd = as.Date(.data$decompEnd, "1970-01-01")
      ) %>%
      mutate(
        liftDays = as.numeric(
          difftime(.data$liftEndDate, .data$liftStartDate, units = "days")
        ),
        decompDays = as.numeric(
          difftime(.data$decompEnd, .data$decompStart, units = "days")
        )
      ) %>%
      mutate(
        decompAbsScaled = .data$pred / .data$decompDays * .data$liftDays,
        decompAbsTotalScaled = .data$pred_total / .data$decompDays * .data$liftDays
      ) %>%
      mutate(
        liftMedia = .data$channel_selected,
        liftStart = .data$liftStartDate,
        liftEnd = .data$liftEndDate,
        mape_lift = abs((.data$decompAbsScaled - .data$liftAbs) / .data$liftAbs),
        calibrated_pct = .data$decompAbsScaled / .data$decompAbsTotalScaled
      ) %>%
      dplyr::select(
        .data$liftMedia, .data$liftStart, .data$liftEnd, .data$liftAbs,
        .data$decompStart, .data$decompEnd, .data$decompAbsScaled,
        .data$decompAbsTotalScaled, .data$calibrated_pct, .data$mape_lift
      )

    return(liftCollect)
  }
}
