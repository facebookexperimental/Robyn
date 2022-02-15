# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

check_conv_error <- function(OutputModels, n_cuts = 10, max_sd = 0.025) {

  # Gather all trials
  for (i in seq_along(OutputModels)) {
    if (i == 1) df <- data.frame()
    temp <- OutputModels[[i]]$resultCollect$resultHypParam %>% mutate(trial = i)
    df <- rbind(df, temp)
  }

  # Calculate deciles
  dt_objfunc_cvg <- tidyr::gather(df, "error_type", "value", any_of(c("nrmse", "decomp.rssd", "mape"))) %>%
    select(ElapsedAccum, trial, error_type, value) %>%
    arrange(trial, ElapsedAccum) %>%
    filter(value > 0) %>%
    mutate(error_type = toupper(error_type)) %>%
    group_by(error_type, trial) %>% mutate(iter = row_number()) %>% ungroup() %>%
    mutate(cuts = cut(
      iter, breaks = seq(0, max(iter), length.out = n_cuts + 1),
      include.lowest = TRUE, ordered_result = TRUE, dig.lab = 6))

  # Calculate sd on each cut to alert user
  errors <- dt_objfunc_cvg %>%
    group_by(error_type, cuts) %>%
    summarise(median = median(value), std = sd(value), .groups = "drop") %>%
    mutate(alert = std > max_sd)
  last_std <- errors %>% group_by(error_type) %>% slice(n_cuts)
  warnings <- NULL
  for (i in seq_along(last_std$error_type)) {
    if (last_std$alert[i]) {
      temp <- sprintf(
        "Error %s hasn't converged yet (quantile-%s: sd %s > %s)",
        last_std$error_type[i], n_cuts, signif(last_std$std[i], 1), max_sd)
      warning(paste("Test with more iterations:", temp))
      warnings <- c(warnings, temp)
    }
  }

  # Elbow plot (like ROI's) looks weird
  # dt_objfunc_cvg %>%
  #   ggplot(aes(x = iter, y = value, colour = as.character(trial), group = error_type)) +
  #   geom_line() +
  #   facet_grid(. ~ error_type, scales = "free") +
  #   theme_lares(legend = "top") +
  #   labs(colour = "Trial", x = "Iterations", y = NULL)

  plot <- dt_objfunc_cvg %>%
    mutate(id = as.integer(cuts)) %>%
    mutate(cuts = factor(cuts, levels = rev(levels(cuts)))) %>%
    ggplot(aes(x = value, y = cuts, fill = -id)) +
    ggridges::geom_density_ridges(
      scale = 2.5, col = "white", quantile_lines = TRUE, quantiles = 2, alpha = 0.7) +
    facet_grid(. ~ error_type, scales = "free") +
    scale_fill_distiller(palette = "GnBu") +
    guides(fill = "none") +
    theme_lares() +
    labs(x = "Errors", y = "Iterations [#]",
         title = "Errors convergence by iterations quantiles",
         subtitle = paste(max(dt_objfunc_cvg$trial), "trials combined"),
         caption = if (!is.null(warnings)) paste(warnings, collapse = "\n") else NULL)

  return(invisible(list(
    plot = plot,
    errors = errors
  )))

}
