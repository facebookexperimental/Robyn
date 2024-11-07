# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn Calibration Function
#'
#' \code{robyn_calibrate()} consumes source of truth or proxy data for
#' saturation or adstock curve estimation.
#'
#' @inheritParams robyn_run
#' @param df_curve_sot data.frame. Requires two columns named spend and response.
#' Recommended sources of truth are Halo R&F or Meta conversion lift.
#' @param curve_type Character. Currently only allows saturation calibration
#' and only supports Hill function. Possible values are \code{c(
#' "saturation_reach", "saturation_revenue", "saturation_conversion")}.
#' @param hp_bounds list. Currently only allows Hill for saturation. Ranges
#' for alpha and gamma are provided as Hill parameters.
#' @param max_trials integer. Different trials have different starting point
#' and provide diversified sampling paths.
#' @param max_iters integer. Loss is minimized while iteration increases.
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
#' # Dummy source of truth data
#' df_curve_sot <- data.frame(
#'   spend = c(0, 1933, 94574, 131815, 370320, 470523, 489839,
#'             514386, 531668, 532889),
#'   response = c(0, 72484, 586912, 749784, 1339424, 1553394, 1593612,
#'             1643194, 1677396, 1679811))
#'
#' # Default hyperparameter ranges for saturation hill function
#' hp_bounds <- list(hill = list(alpha = c(0, 10), gamma = c(0,1)))
#'
#' curve_out <- robyn_calibrate(
#'   df_curve_sot = df_curve_sot,
#'   hp_bounds = hp_bounds,
#'   max_trials = 5
#' )
#' @return List. Class: \code{curve_out}. Contains the results of all trials
#' and iterations modeled.
#' @export
robyn_calibrate <- function(
    df_curve_sot = NULL,
    curve_type = "saturation_reach",
    hp_bounds = list(
      hill = list(
        alpha = c(0, 10),
        gamma = c(0,1))),
    max_trials = 10,
    max_iters = 2500,
    loss_min_step_rel = 0.01,
    loss_stop_rel = 0.05,
    burn_in_rel = 0.1,
    sim_n = 50,
    hp_interval = 0.95,
    quiet = FALSE,
    ...) {

  ## check all inputs
  # df_curve_sot df format
  # curve types
  # hp_bounds format
  # hp_interval

  if (grepl("saturation", curve_type)) {

    response_sot <- df_curve_sot$response
    spend_sot <- df_curve_sot$spend

    ## get hyperparameter bounds
    if (is.null(hp_bounds)) {
      hp_bounds <- list(hill = list(alpha = c(0, 10), gamma = c(0,1)))
    }
    hp_bounds_loop <- hp_bounds[["hill"]]

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

      my_tuple <- reticulate::tuple(as.integer(2))
      instrumentation <- ng$p$Array(shape = my_tuple, lower = 0, upper = 1)
      optimizer <- ng$optimizers$registry["TwoPointsDE"](instrumentation, budget = max_iters)

      ## inner while loop that stops when converged
      ng_hp_i <- list()
      loss_collect_i <- c()
      pred_collect_i <- list()
      if (!quiet) pb_cf <- txtProgressBar(min = 0, max = max_iters_vec[j], style = 3)
      loop_continue <- TRUE
      i = 0

      while (loop_continue) {
        i <- i +1
        if (!quiet) setTxtProgressBar(pb_cf, i)

        ## Nevergrad ask sample
        ng_hp_i[[i]] <- optimizer$ask()
        ng_hp_val <- ng_hp_i[[i]]$value
        ng_hp_val_scaled <- mapply(function(hpb, hp) {
          qunif(hp, min = min(hpb), max = max(hpb))
        },
        hpb = hp_bounds_loop,
        hp = ng_hp_val)
        alpha <- ng_hp_val_scaled["alpha"]
        gamma <- ng_hp_val_scaled["gamma"]

        ## predict saturation vector
        response_pred <-  saturation_hill(spend_sot, alpha, gamma)[["x_saturated"]]
        response_sot_scaled <- .min_max_norm(response_sot)

        ## get loss
        loss_iter <- .mse_loss(y = response_sot_scaled, y_hat = response_pred)
        max_loss <- ifelse(i==1, loss_iter, max(max_loss, loss_iter))
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
                " iterations. Increase iterations or adjust convergence criterias."))
            }
          } else {
            current_unit <- (i-loss_stop_abs+1):i
            previous_unit <- current_unit-loss_stop_abs
            loss_unit_change <- (mean(loss_collect_i[current_unit]) - mean(loss_collect_i[previous_unit]))
            loop_continue <- !all(loss_unit_change > 0, loss_unit_change <= loss_min_step_abs)

            if (loop_continue == FALSE) {
              if (!quiet) {
                close(pb_cf)
                message(paste0(
                  "Trial ", j, " converged & stopped at iteration ", i,
                  " from ", max_iters_vec[j]))
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

    ## saturation hill

    if (TRUE) {
      best_alpha <- .dot_product(hp_bounds_loop[["alpha"]], best_hp[1])
      best_gamma <- .dot_product(hp_bounds_loop[["gamma"]], best_hp[2])
      #best_response_pred <- saturation_hill(spend_sot, best_alpha, best_gamma)[["x_saturated"]]
      #best_inflexion <- saturation_hill(spend_sot, best_alpha, best_gamma)[["inflexion"]]
      alpha_collect <- lapply(ng_hp, FUN = function(x) {
        sapply(x, FUN = function(y) .dot_product(hp_bounds_loop[["alpha"]], y$value[1]))
      })
      gamma_collect <- lapply(ng_hp, FUN = function(x) {
        sapply(x, FUN = function(y) .dot_product(hp_bounds_loop[["gamma"]], y$value[2]))
      })

      ## slice by convergence
      burn_in_abs <- rep(max_iters * burn_in_rel, max_trials)
      alpha_collect_converged <- unlist(mapply(
        function(x, start, end) x[start:end],
        x = alpha_collect, start = burn_in_abs,
        end = max_iters_vec, SIMPLIFY = FALSE))
      gamma_collect_converged <- unlist(mapply(
        function(x, start, end) x[start:end],
        x = gamma_collect, start = burn_in_abs,
        end = max_iters_vec, SIMPLIFY = FALSE))

      ## get calibration range for hyparameters
      qt_alpha_out <- .qti(x = alpha_collect_converged, interval = hp_interval)
      qt_gamma_out <- .qti(x = gamma_collect_converged, interval = hp_interval)
    }

    ## plotting & prompting
    df_sot_plot <- data.frame(spend = spend_sot, response = response_sot_scaled)
    temp_spend <- seq(0, max(spend_sot), by = sim_n)
    temp_sat <- saturation_hill(temp_spend, best_alpha, best_gamma)[["x_saturated"]]
    df_pred_plot <- data.frame(spend = temp_spend, response = temp_sat)

    sim_alphas <- alpha_collect_converged[
      alpha_collect_converged > qt_alpha_out[1] &
        alpha_collect_converged < qt_alpha_out[2]]
    sim_alphas <- sample(sim_alphas, sim_n, replace = TRUE)
    sim_gammas <- gamma_collect_converged[
      gamma_collect_converged > qt_gamma_out[1] &
        gamma_collect_converged < qt_gamma_out[2]]
    sim_gammas <- sample(sim_gammas, sim_n, replace = TRUE)

    # simulation for plotting
    sim_collect <- list()
    for (i in 1:sim_n) {
      sim_collect[[i]] <- saturation_hill(temp_spend, sim_alphas[i], sim_gammas[i])[["x_saturated"]]
    }
    sim_collect <- data.frame(
      sim = as.character(c(sapply(1:sim_n, function(x) rep(x, length(temp_spend))))),
      sim_spend = rep(temp_spend, sim_n),
      sim_saturation = unlist(sim_collect))

    y_lab <- stringr::str_to_title(gsub("saturation_", "", curve_type))
    p_lines <- ggplot() +
      geom_line(data = sim_collect,
                aes(x = .data$sim_spend, y = .data$sim_saturation,
                    color = .data$sim), size = 2, alpha = 0.2) +
      scale_colour_grey() +
      theme_lares(legend = "none", ...) +
      geom_point(
        data = df_sot_plot,
        aes(x=.data$spend, y=.data$response)) +
      geom_line(
        data = df_pred_plot,
        aes(x=.data$spend, y=.data$response), color = "blue") +
      labs(title = paste0("Spend to ", y_lab, " saturation curve estimation")) +
      ylab(y_lab) + xlab("Spend")

    df_mse <- data.frame(mse = unlist(loss_collect),
                         iterations = unlist(mapply(function(x) 1:x, max_iters_vec, SIMPLIFY = FALSE)),
                         trials = as.character(unlist(
                           mapply(function (x, y) rep(x, y),
                                  x = 1:max_trials, y = max_iters_vec))))
    p_mse <- df_mse %>%
      mutate(trials = factor(.data$trials, levels = seq(max_trials))) %>%
      ggplot(aes(x = .data$iterations, y = .data$mse)) +
      geom_line(size = 0.2) +
      facet_grid(.data$trials ~ .) +
      labs(title = paste0("Loss convergence with error reduction of ",
                          round((1 - best_loss_val / max_loss), 4) * 100, "%"),
           x = "Iterations", y = "MSE") +
      theme_lares(grid = "Xx", ...) + scale_x_abbr() +
      theme(axis.title.y = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.y = element_blank())

    p_alpha <- data.frame(alpha = alpha_collect_converged) %>% ggplot(aes(x = alpha)) +
      geom_density(fill = "grey99", color = "grey") +
      labs(title = paste0("Alpha (Hill) density after ", round(burn_in_rel * 100),"% burn-in"),
           subtitle = paste0("95% interval: ", round(qt_alpha_out[1],4), "-", round(qt_alpha_out[2],4))) +
      theme_lares(...) + scale_y_abbr()
    p_alpha <- geom_density_ci(p_alpha, qt_alpha_out[1], qt_alpha_out[2], fill = "lightblue")
    p_gamma <- data.frame(gamma = gamma_collect_converged) %>% ggplot(aes(x = gamma)) +
      geom_density(fill = "grey99", color = "grey") +
      labs(title = paste0("Gamma (Hill) density after ", round(burn_in_rel * 100),"% burn-in"),
           subtitle = paste0("95% interval: ", round(qt_gamma_out[1],4), "-", round(qt_gamma_out[2],4))) +
      theme_lares(...) + scale_y_abbr()
    p_gamma <- geom_density_ci(p_gamma, qt_gamma_out[1], qt_gamma_out[2], fill = "lightblue")

    if (!quiet) message(
      paste0("\nBest alpha: ", round(best_alpha,4), " (",
             paste0(round(qt_alpha_out,4), collapse = "-"), ")",

             ", Best gamma: ", round(best_gamma,4), " (",
             paste0(round(qt_gamma_out,4), collapse = "-"), ")",
             ", Max spend: ", max(spend_sot)))

    curve_out <- list(
      hill = list(alpha = c(qt_alpha_out), gamma = c(qt_gamma_out)),
      plot = p_lines / p_mse / (p_alpha + p_gamma) +
        plot_annotation(
          theme = theme_lares(background = "white", ...)
        )
    )
    return(curve_out)
  }
}


lift_calibration <- function(calibration_input,
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
    split_channels <- strsplit(calibration_input$channel, split = "\\+")

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
          if (adstock == "weibull_pdf") {
            m_imme <- x_list$x_imme
          } else {
            m_imme <- m
          }
          m_total <- x_list$x_decayed
          m_caov <- m_total - m_imme

          ## 2. Saturation
          m_caov_calib <- m_caov[calib_pos]
          m_total_rw <- m_total[wind_start:wind_end]
          alpha <- hypParamSam[paste0(get_channels[l_chn], "_alphas")][[1]][[1]]
          gamma <- hypParamSam[paste0(get_channels[l_chn], "_gammas")][[1]][[1]]
          m_calib_caov_sat <- saturation_hill(
            m_total_rw,
            alpha = alpha, gamma = gamma, x_marginal = m_caov_calib
          )
          m_calib_caov_decomp <- m_calib_caov_sat * coefs$s0[coefs$rn == get_channels[l_chn]]
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
        liftMedia = .data$channel,
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
