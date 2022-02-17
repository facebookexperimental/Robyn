# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

check_conv_error <- function(OutputModels, n_cuts = 10, threshold_sd = 0.025) {

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

  # Calculate sd on each cut to alert user
  errors <- dt_objfunc_cvg %>%
    group_by(.data$error_type, .data$cuts) %>%
    summarise(
      median = median(.data$value),
      std = sd(.data$value),
      .groups = "drop"
    ) %>%
    mutate(
      med_var_P = abs(round(100 * (.data$median - lag(.data$median)) / .data$median, 2)),
      alert = .data$std > threshold_sd
    )
  last_std <- errors %>%
    group_by(.data$error_type) %>%
    slice(n_cuts)
  conv_msg <- NULL
  for (i in seq_along(last_std$error_type)) {
    if (last_std$alert[i]) {
      temp <- sprintf(
        "Obj.func. %s hasn't converged (qt-%s sd: %s > %s threshold) -> More iterations recommended",
        last_std$error_type[i], n_cuts, signif(last_std$std[i], 1), threshold_sd
      )
    } else {
      temp <- sprintf(
        "Obj.func. %s has converged (qt-%s sd: %s <= %s threshold)",
        last_std$error_type[i], n_cuts, signif(last_std$std[i], 1), threshold_sd
      )
    }
    message(temp)
    conv_msg <- c(conv_msg, temp)
  }

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
      x = "Errors", y = "Iterations [#]",
      title = "Errors convergence by iterations quantiles",
      subtitle = paste(max(dt_objfunc_cvg$trial), "trials combined"),
      caption = paste(conv_msg, collapse = "\n")
    )

  moo_cloud_plot <- ggplot(df, aes(x = .data$nrmse, y = .data$decomp.rssd, colour = .data$ElapsedAccum)) +
    scale_colour_gradient(low = "skyblue", high = "navyblue") +
    labs(
      title = ifelse(!calibrated, "Multi-objective evolutionary performance",
                     "Multi-objective evolutionary performance with calibration"
      ),
      subtitle = sprintf("%s trials with %s iterations each",
                         max(df$trial), max(dt_objfunc_cvg$cuts)),
      x = "NRMSE",
      y = "DECOMP.RSSD",
      colour = "Time [s]",
      size = "Mape",
      alpha = NULL,
      caption = paste(conv_msg, collapse = "\n")
    ) +
    theme_lares()
  # facet_wrap(.data$trial~.)
  if(calibrated) {
    moo_cloud_plot <- moo_cloud_plot + geom_point(data = df, aes(size = mape, alpha = 1-mape), )
  } else {
    moo_cloud_plot <- moo_cloud_plot + geom_point()
  }

  return(invisible(list(
    moo_distrb_plot = moo_distrb_plot,
    moo_cloud_plot = moo_cloud_plot,
    errors = select(errors, -.data$alert)
  )))
}
