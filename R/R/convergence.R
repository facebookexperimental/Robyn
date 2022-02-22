# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Check model convergence
#'
#' The \code{robyn_converge()} function consumes output from \code{robyn_run()}
#' and calculate convergence status and output convergence plots.
#'
#' @param OutputModels List. Output from \code{robyn_run()}
#' @param n_cuts Integer. Default to 20 (last 5%). Convergence is calculated
#' on last quantile of cuts.
#' @param threshold_sd Numeric. Default to 0.025 that is empirically derived.
#' @examples
#' \dontrun{
#' OutputModels <- robyn_converge(
#'   OutputModels = OutputModels,
#'   n_cuts = 10,
#'   threshold_sd = 0.025
#' )
#' }
#' @export
robyn_converge <- function(OutputModels, n_cuts = 20, threshold_sd = 0.025) {

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
    filter(.data$value > 0) %>%
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

  # Calculate sd and median on each cut to alert user on:
  # 1) last quantile's sd < threshold_sd
  # 2) last quantile's median < first quantile's median - 2 * sd
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
      med_var_P = abs(round(100 * (.data$median - lag(.data$median)) / .data$median, 2)),
      flag_sd = .data$std > threshold_sd
    ) %>%
    group_by(.data$error_type) %>%
    mutate(flag_med = dplyr::last(.data$median[1]) < dplyr::first(.data$median[2]) - 2 * dplyr::first(.data$std))

  conv_msg <- NULL
  for (obj_fun in unique(errors$error_type)) {
    temp.df <- filter(errors, .data$error_type == obj_fun) %>%
      mutate(median = signif(median, 2))
    last.qt <- tail(temp.df, 1)
    temp <- glued(paste(
        "{error_type} {did}converged: sd {sd} @qt.{quantile} {symb_sd} {sd_threh} &",
        "med {qtn_median} @qt.{quantile} {symb_med} {med_threh} med@qt.1-2*sd"),
        error_type = last.qt$error_type,
        did = ifelse(last.qt$flag_sd | last.qt$flag_med, "NOT ", ""),
        sd = signif(last.qt$std, 1),
        symb_sd = ifelse(last.qt$flag_sd, ">", "<="),
        sd_threh = threshold_sd,
        quantile = round(100/n_cuts),
        qtn_median = temp.df$median[n_cuts],
        symb_med = ifelse(last.qt$flag_med, ">", "<="),
        med_threh = signif(temp.df$median[1] - 2 * temp.df$std[1], 2)
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
      subtitle = paste(max(dt_objfunc_cvg$trial), "trials combined"),
      caption = paste(conv_msg, collapse = "\n")
    )

  moo_cloud_plot <- ggplot(df, aes(
    x = .data$nrmse, y = .data$decomp.rssd, colour = .data$ElapsedAccum)) +
    scale_colour_gradient(low = "skyblue", high = "navyblue") +
    labs(
      title = ifelse(!calibrated, "Multi-objective evolutionary performance",
        "Multi-objective evolutionary performance with calibration"
      ),
      subtitle = sprintf(
        "%s trial%s with %s iterations each",
        max(df$trial), ifelse(max(df$trial) > 1, "s", ""), max(dt_objfunc_cvg$cuts)
      ),
      x = "NRMSE",
      y = "DECOMP.RSSD",
      colour = "Time [s]",
      size = "MAPE",
      alpha = NULL,
      caption = paste(conv_msg, collapse = "\n")
    ) +
    theme_lares()

  if (calibrated) {
    moo_cloud_plot <- moo_cloud_plot + geom_point(data = df, aes(size = .data$mape, alpha = 1 - .data$mape))
  } else {
    moo_cloud_plot <- moo_cloud_plot + geom_point()
  }

  cvg_out <- list(
    moo_distrb_plot = moo_distrb_plot,
    moo_cloud_plot = moo_cloud_plot,
    errors = errors,
    conv_msg = conv_msg
  )
  attr(cvg_out, "threshold_sd") <- threshold_sd

  return(invisible(cvg_out))
}
