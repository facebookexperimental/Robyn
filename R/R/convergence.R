# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Check Models Convergence
#'
#' \code{robyn_converge()} consumes \code{robyn_run()} outputs
#' and calculate convergence status and builds convergence plots.
#' Convergence is calculated by default using the following criteria
#' (having kept the default parameters: sd_qtref = 3 and med_lowb = 2):
#' \describe{
#'   \item{Criteria #1:}{Last quantile's standard deviation < first 3
#'   quantiles' mean standard deviation}
#'   \item{Criteria #2:}{Last quantile's absolute median < absolute first
#'   quantile's absolute median - 2 * first 3 quantiles' mean standard
#'   deviation}
#' }
#' Both mentioned criteria have to be satisfied to consider MOO convergence.
#'
#' @param OutputModels List. Output from \code{robyn_run()}.
#' @param n_cuts Integer. Default to 20 (5\% cuts each).
#' @param sd_qtref Integer. Reference quantile of the error convergence rule
#' for standard deviation (Criteria #1). Defaults to 3.
#' @param med_lowb Integer. Lower bound distance of the error convergence rule
#' for median. (Criteria #2). Default to 3.
#' @param nrmse_win Numeric vector. Lower and upper quantiles thresholds to
#' winsorize NRMSE. Set values within [0,1]; default: c(0, 0.998) which is 1/500.
#' @param ... Additional parameters
#' @examples
#' \dontrun{
#' # Having OutputModels results
#' MOO <- robyn_converge(
#'   OutputModels,
#'   n_cuts = 10,
#'   sd_qtref = 3,
#'   med_lowb = 3
#' )
#' }
#' @return List. Plots and MOO convergence results.
#' @export
robyn_converge <- function(OutputModels,
                           n_cuts = 20, sd_qtref = 3, med_lowb = 2,
                           nrmse_win = c(0, 0.998), ...) {
  stopifnot(n_cuts > min(c(sd_qtref, med_lowb)) + 1)

  # Gather all trials
  get_trials <- which(names(OutputModels) %in% paste0("trial", seq(OutputModels$trials)))
  df <- bind_rows(lapply(OutputModels[get_trials], function(x) x$resultCollect$resultHypParam))
  calibrated <- isTRUE(sum(df$mape) > 0)

  # Calculate deciles
  dt_objfunc_cvg <- tidyr::gather(df, "error_type", "value", any_of(c("nrmse", "decomp.rssd", "mape"))) %>%
    select(.data$ElapsedAccum, .data$trial, .data$error_type, .data$value) %>%
    arrange(.data$trial, .data$ElapsedAccum) %>%
    filter(.data$value > 0, is.finite(.data$value)) %>%
    mutate(error_type = toupper(.data$error_type)) %>%
    group_by(.data$error_type, .data$trial) %>%
    mutate(iter = row_number()) %>%
    ungroup() %>%
    mutate(cuts = cut(
      .data$iter,
      breaks = seq(0, max(.data$iter), length.out = n_cuts + 1),
      labels = round(seq(max(.data$iter) / n_cuts, max(.data$iter), length.out = n_cuts)),
      include.lowest = TRUE, ordered_result = TRUE, dig.lab = 6
    ))

  # Calculate standard deviations and absolute medians on each cut
  errors <- dt_objfunc_cvg %>%
    group_by(.data$error_type, .data$cuts) %>%
    summarise(
      n = n(),
      median = median(.data$value),
      std = sd(.data$value),
      .groups = "drop"
    ) %>%
    group_by(.data$error_type) %>%
    mutate(
      med_var_P = abs(round(100 * (.data$median - lag(.data$median)) / .data$median, 2))
    ) %>%
    group_by(.data$error_type) %>%
    mutate(
      first_med = abs(dplyr::first(.data$median)),
      first_med_avg = abs(mean(.data$median[1:sd_qtref])),
      last_med = abs(dplyr::last(.data$median)),
      first_sd = dplyr::first(.data$std),
      first_sd_avg = mean(.data$std[1:sd_qtref]),
      last_sd = dplyr::last(.data$std)
    ) %>%
    mutate(
      med_thres = abs(.data$first_med - med_lowb * .data$first_sd_avg),
      flag_med = abs(.data$median) < .data$med_thres,
      flag_sd = .data$std < .data$first_sd_avg
    )

  conv_msg <- NULL
  for (obj_fun in unique(errors$error_type)) {
    temp.df <- filter(errors, .data$error_type == obj_fun) %>%
      mutate(median = signif(median, 2))
    last.qt <- tail(temp.df, 1)
    greater <- ">" # intToUtf8(8814)
    temp <- glued(
      paste(
        "{error_type} {did}converged: sd@qt.{quantile} {sd} {symb_sd} {sd_threh} &",
        "|med@qt.{quantile}| {qtn_median} {symb_med} {med_threh}"
      ),
      error_type = last.qt$error_type,
      did = ifelse(last.qt$flag_sd & last.qt$flag_med, "", "NOT "),
      sd = signif(last.qt$last_sd, 2),
      symb_sd = ifelse(last.qt$flag_sd, "<=", greater),
      sd_threh = signif(last.qt$first_sd_avg, 2),
      quantile = n_cuts,
      qtn_median = signif(last.qt$last_med, 2),
      symb_med = ifelse(last.qt$flag_med, "<=", greater),
      med_threh = signif(last.qt$med_thres, 2)
    )
    conv_msg <- c(conv_msg, temp)
  }
  message(paste(paste("-", conv_msg), collapse = "\n"))

  subtitle <- sprintf(
    "%s trial%s with %s iterations%s using %s",
    max(df$trial), ifelse(max(df$trial) > 1, "s", ""), max(dt_objfunc_cvg$cuts),
    ifelse(max(df$trial) > 1, " each", ""), OutputModels$nevergrad_algo
  )

  moo_distrb_plot <- dt_objfunc_cvg %>%
    mutate(id = as.integer(.data$cuts)) %>%
    mutate(cuts = factor(.data$cuts, levels = rev(levels(.data$cuts)))) %>%
    group_by(.data$error_type) %>%
    mutate(value = lares::winsorize(.data$value, nrmse_win)) %>%
    ggplot(aes(x = .data$value, y = .data$cuts, fill = -.data$id)) +
    ggridges::geom_density_ridges(
      scale = 2.5, col = "white", quantile_lines = TRUE, quantiles = 2, alpha = 0.7
    ) +
    facet_grid(. ~ .data$error_type, scales = "free") +
    scale_fill_distiller(palette = "GnBu") +
    guides(fill = "none") +
    theme_lares() +
    labs(
      x = "Objective functions", y = "Iterations [#]",
      title = "Objective convergence by iterations quantiles",
      subtitle = subtitle,
      caption = paste(conv_msg, collapse = "\n")
    )

  moo_cloud_plot <- df %>%
    mutate(nrmse = lares::winsorize(.data$nrmse, nrmse_win)) %>%
    ggplot(aes(
    x = .data$nrmse, y = .data$decomp.rssd, colour = .data$ElapsedAccum
  )) +
    scale_colour_gradient(low = "skyblue", high = "navyblue") +
    labs(
      title = ifelse(!calibrated, "Multi-objective evolutionary performance",
        "Multi-objective evolutionary performance with calibration"
      ),
      subtitle = subtitle,
      x = ifelse(max(nrmse_win) == 1, "NRMSE", sprintf("NRMSE [Winsorized %s]", paste(nrmse_win, collapse = "-"))),
      y = "DECOMP.RSSD",
      colour = "Time [s]",
      size = "MAPE",
      alpha = NULL,
      caption = paste(conv_msg, collapse = "\n")
    ) +
    theme_lares()

  if (calibrated) {
    moo_cloud_plot <- moo_cloud_plot +
      geom_point(data = df, aes(size = .data$mape, alpha = 1 - .data$mape)) +
      guides(alpha = "none")
  } else {
    moo_cloud_plot <- moo_cloud_plot + geom_point()
  }

  cvg_out <- list(
    moo_distrb_plot = moo_distrb_plot,
    moo_cloud_plot = moo_cloud_plot,
    errors = errors,
    conv_msg = conv_msg
  )
  attr(cvg_out, "sd_qtref") <- sd_qtref
  attr(cvg_out, "med_lowb") <- med_lowb

  return(invisible(cvg_out))
}

test_cvg <- function() {
  # Experiment with gamma distribution fitting
  gamma_mle <- function(params, x) {
    gamma_shape <- params[[1]]
    gamma_scale <- params[[2]]
    # Negative log-likelihood
    return(-sum(dgamma(x, shape = gamma_shape, scale = gamma_scale, log = TRUE)))
  }
  f_geo <- function(a, r, n) {
    for (i in 2:n) a[i] <- a[i - 1] * r
    return(a)
  }
  seq_nrmse <- f_geo(5, 0.7, 100)
  df_nrmse <- data.frame(x = 1:100, y = seq_nrmse, type = "true")
  mod_gamma <- nloptr(
    x0 = c(1, 1), eval_f = gamma_mle, lb = c(0, 0),
    x = seq_nrmse,
    opts = list(algorithm = "NLOPT_LN_SBPLX", maxeval = 1e5)
  )
  gamma_params <- mod_gamma$solution
  seq_nrmse_gam <- 1 / dgamma(seq_nrmse, shape = gamma_params[[1]], scale = gamma_params[[2]])
  seq_nrmse_gam <- seq_nrmse_gam / (max(seq_nrmse_gam) - min(seq_nrmse_gam))
  seq_nrmse_gam <- max(seq_nrmse) * seq_nrmse_gam
  range(seq_nrmse_gam)
  range(seq_nrmse)
  df_nrmse_gam <- data.frame(x = 1:100, y = seq_nrmse_gam, type = "pred")
  df_nrmse <- bind_rows(df_nrmse, df_nrmse_gam)
  p <- ggplot(df_nrmse, aes(.data$x, .data$y, color = .data$type)) +
    geom_line()
  return(p)
  # g_low = qgamma(0.025, shape=gamma_params[[1]], scale= gamma_params[[2]])
  # g_up = qgamma(0.975, shape=gamma_params[[1]], scale= gamma_params[[2]])
}
