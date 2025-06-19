#' Placebo test by shuffling one channel, re-running Robyn, 
#' and exporting only the placebo chart
#'
#' @param OutputCollect A robyn_outputs() object (after your base fit)
#' @param channel      Character: which paid media channel to shuffle
#' @param plot_folder  (optional) where to save "placebo_test.png". 
#'                     Defaults to OutputCollect$plot_folder.
#' @param export       Logical, default TRUE. If TRUE, writes the PNG; 
#'                     if FALSE, just returns the ggplot.
#' @return             The same OutputCollect (now with `$placebo`), 
#'                     and (invisibly) the ggplot object.
#' @importFrom stats t.test var.test
#' @export
robyn_placebo <- function(OutputCollect,
                          channel,
                          plot_folder = NULL,
                          export      = TRUE) {
  InputCollect  <- OutputCollect$InputCollect
  OutputModels  <- OutputCollect$OutputModels

  # 1) Shuffle the channel
  dt_shuf <- data.table::copy(InputCollect$dt_input)
  dt_shuf[[channel]] <- sample(dt_shuf[[channel]])

  # 2) Rebuild inputs for shuffled data
  ic_shuf <- robyn_inputs(
    dt_input          = dt_shuf,
    dt_holidays       = InputCollect$dt_holidays,
    date_var          = InputCollect$date_var,
    dep_var           = InputCollect$dep_var,
    dep_var_type      = InputCollect$dep_var_type,
    prophet_vars      = InputCollect$prophet_vars,
    prophet_country   = InputCollect$prophet_country,
    context_vars      = InputCollect$context_vars,
    paid_media_spends = InputCollect$paid_media_spends,
    paid_media_vars   = InputCollect$paid_media_vars,
    organic_vars      = InputCollect$organic_vars,
    factor_vars       = InputCollect$factor_vars,
    window_start      = InputCollect$window_start,
    window_end        = InputCollect$window_end,
    adstock           = InputCollect$adstock,
    hyperparameters   = InputCollect$hyperparameters
  )

  # 3) Re-run Robyn on that shuffled input (export = FALSE so no other files get written)
  om_shuf <- robyn_run(
    InputCollect = ic_shuf,
    iterations   = OutputModels$iterations,
    trials       = OutputModels$trials,
    ts_validation= OutputModels$ts_validation,
    export       = FALSE
  )

  # 4) Collect NRMSE distributions (no need to run robyn_outputs on the shuffled run)
  get_nrmse <- function(om) {
    do.call(rbind, lapply(seq_len(om$trials), function(i) {
      trial_i <- om[[paste0("trial", i)]]
      trial_i$resultCollect$resultHypParam
    }))$nrmse
  }
  orig_dist <- get_nrmse(OutputModels)
  sham_dist <- get_nrmse(om_shuf)

  # 5) Run t-test and F-test
  t_out <- t.test(sham_dist, orig_dist, alternative = "greater")
  f_out <- var.test(sham_dist, orig_dist, alternative = "greater")

  # 6) Store everything back into OutputCollect$placebo
  OutputCollect$placebo <- list(
    channel   = channel,
    orig_dist = orig_dist,
    sham_dist = sham_dist,
    t_test    = t_out,
    f_test    = f_out
  )

  # 7) Build the placebo plot (density + violin) using plot_placebo()
  pPlacebo <- plot_placebo(OutputCollect)

  # 8) Save only “placebo_test.png” (no other charts)
  if (is.null(plot_folder)) {
    plot_folder <- OutputCollect$plot_folder
  }
  if (export) {
    if (!dir.exists(plot_folder)) {
      dir.create(plot_folder, recursive = TRUE)
    }
    ggplot2::ggsave(
      filename = file.path(plot_folder, paste0("placebo_test_", channel, ".png")),
      plot     = pPlacebo,
      dpi      = 600,
      width    = 12,
      height   = 8
    )
  }

  invisible(pPlacebo)   # return the ggplot invisibly
  return(OutputCollect) # but still return the new OutputCollect object
}
