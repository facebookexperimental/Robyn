# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Check Models Convergence
#'
#' \code{robyn_converge()} consumes \code{robyn_run()} outputs
#' and calculate convergence status and builds convergence plots.
#' Convergence is calculated by default using the following criteria
#' (having kept the default parameters: sd_qtref = 3 and med_lowb = 3):
#' \describe{
#'   \item{Criteria #1:}{Last quantile's standard deviation < first 3
#'   quantiles' mean standard deviation}
#'   \item{Criteria #2:}{Last quantile's median < first quantile's
#'   median - 3 * first 3 quantiles' mean standard deviation.}
#' }
#' Both mentioned criteria have to be satisfied to consider MOO convergence.
#'
#' @param OutputModels List. Output from \code{robyn_run()}.
#' @param n_cuts Integer. Default to 20 (5\% cuts each).
#' @param sd_qtref Integer. Reference quantile of the error convergence rule
#' for standard deviation (Criteria #1). Defaults to 3.
#' @param med_lowb Integer. Lower bound distance of the error convergence rule
#' for median. (Criteria #2). Default to 3.
#' @param ... Additional parameters
#' @examples
#' \dontrun{
#' OutputModels <- robyn_converge(
#'   OutputModels = OutputModels,
#'   n_cuts = 10,
#'   sd_qtref = 3,
#'   med_lowb = 3
#' )
#' }
#' @export
robyn_converge <- function(OutputModels, n_cuts = 20, sd_qtref = 3, med_lowb = 3, ...) {

  stopifnot(n_cuts > min(c(sd_qtref, med_lowb)) + 1)

  # Gather all trials
  get_lists <- as.logical(grepl("trial", names(OutputModels)) * sapply(OutputModels, is.list))
  OutModels <- OutputModels[get_lists]
  for (i in seq_along(OutModels)) {
    if (i == 1) df <- data.frame()
    temp <- OutModels[[i]]$resultCollect$resultHypParam %>% mutate(trial = i)
    df <- rbind(df, temp)
  }
  calibrated <- sum(df$mape) > 0

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

  # Calculate sd and median on each cut to alert user when no convergence
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
    mutate(first_med = dplyr::first(.data$median),
           first_med_avg = mean(.data$median[1:sd_qtref]),
           last_med = dplyr::last(.data$median),
           first_sd = dplyr::first(.data$std),
           first_sd_avg = mean(.data$std[1:sd_qtref]),
           last_sd = dplyr::last(.data$std))  %>%
    mutate(med_thres = .data$first_med - med_lowb * .data$first_sd_avg,
           flag_med = .data$median < .data$first_med - med_lowb * .data$first_sd_avg,
           flag_sd = .data$std < .data$first_sd_avg)

  conv_msg <- NULL
  for (obj_fun in unique(errors$error_type)) {
    temp.df <- filter(errors, .data$error_type == obj_fun) %>%
      mutate(median = signif(median, 2))
    last.qt <- tail(temp.df, 1)
    greater <- ">" #intToUtf8(8814)
    temp <- glued(paste(
        "{error_type} {did}converged: sd@qt.{quantile} {sd} {symb_sd} {sd_threh} &",
        "med@qt.{quantile} {qtn_median} {symb_med} {med_threh} med@qt.1-{med_lowb}*sd"),
        error_type = last.qt$error_type,
        did = ifelse(last.qt$flag_sd & last.qt$flag_med, "", "NOT "),
        sd = signif(last.qt$last_sd, 2),
        symb_sd = ifelse(last.qt$flag_sd, "<=", greater),
        sd_threh = signif(last.qt$first_sd_avg, 2),
        quantile = n_cuts,
        qtn_median = signif(last.qt$last_med, 2),
        symb_med = ifelse(last.qt$flag_med, "<=", greater),
        med_threh = signif(last.qt$med_thres, 2),
        med_lowb = med_lowb
      )
    conv_msg <- c(conv_msg, temp)
  }
  message(paste(paste("-", conv_msg), collapse = "\n"))

  # # Moving average
  # dt_objfunc_cvg %>%
  #   group_by(trial, error_type) %>%
  #   mutate(value_ma = zoo::rollapply(value, 50, mean, fill = NA)) %>%
  #   ggplot(aes(x = iter, y = value_ma, colour = as.character(trial), group = error_type)) +
  #   geom_line(na.rm = TRUE) +
  #   facet_grid(. ~ error_type, scales = "free") +
  #   theme_lares(legend = "top") +
  #   labs(colour = "Trial", x = "Iterations", y = NULL)

  # # Elbow plot (like ROI's) looks weird
  # dt_objfunc_cvg %>%
  #   ggplot(aes(x = iter, y = value, colour = as.character(trial), group = error_type)) +
  #   geom_line() +
  #   facet_grid(. ~ error_type, scales = "free") +
  #   theme_lares(legend = "top") +
  #   labs(colour = "Trial", x = "Iterations", y = NULL)

  subtitle <- sprintf(
    "%s trial%s with %s iterations %s",
    max(df$trial), ifelse(max(df$trial) > 1, "s", ""), max(dt_objfunc_cvg$cuts),
    ifelse(max(df$trial) > 1, "each", "")
  )

  moo_distrb_plot <- dt_objfunc_cvg %>%
    mutate(id = as.integer(.data$cuts)) %>%
    mutate(cuts = factor(.data$cuts, levels = rev(levels(.data$cuts)))) %>%
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

  moo_cloud_plot <- ggplot(df, aes(
    x = .data$nrmse, y = .data$decomp.rssd, colour = .data$ElapsedAccum)) +
    scale_colour_gradient(low = "skyblue", high = "navyblue") +
    labs(
      title = ifelse(!calibrated, "Multi-objective evolutionary performance",
        "Multi-objective evolutionary performance with calibration"
      ),
      subtitle = subtitle,
      x = "NRMSE",
      y = "DECOMP.RSSD",
      colour = "Time [s]",
      size = "MAPE",
      alpha = NULL,
      caption = paste(conv_msg, collapse = "\n")
    ) +
    theme_lares()

  if (calibrated) {
    moo_cloud_plot <- moo_cloud_plot + geom_point(data = df, aes(size = .data$mape, alpha = 1 - .data$mape)) +
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
