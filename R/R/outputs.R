# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Output results into local files
#'
#' Pack \code{robyn_plots()}, \code{robyn_csv()}, and \code{robyn_clusters()}
#' outcomes for \code{robyn_run()} results. When \code{UI=TRUE}, enriched
#' \code{OutputModels} results with additional plots and objects.
#'
#' @param InputCollect,OutputModels \code{robyn_run()} outcomes.
#' @param pareto_fronts Integer. Number of Pareto fronts for the output.
#' \code{pareto_fronts = 1} returns the best models trading off \code{NRMSE} &
#' \code{DECOMP.RSSD}. Increase \code{pareto_fronts} to get more model choices.
#' @param calibration_constraint Numeric. Default to 0.1 and allows 0.01-0.1. When
#' calibrating, 0.1 means top 10% calibrated models are used for pareto-optimal
#' selection. Lower \code{calibration_constraint} increases calibration accuracy.
#' @param plot_folder Character. Path for saving plots. Default
#' to \code{robyn_object} and saves plot in the same directory as \code{robyn_object}.
#' @param plot_folder_sub Character. Customize sub path to save plots. The total
#' path is created with \code{dir.create(file.path(plot_folder, plot_folder_sub))}.
#' For example, plot_folder_sub = "sub_dir".
#' @param plot_pareto Boolean. Set to \code{FALSE} to deactivate plotting
#' and saving model one-pagers. Used when testing models.
#' @param clusters Boolean. Apply \code{robyn_clusters()} to output models?
#' @param selected Character vector. Which models (by \code{solID}) do you
#' wish to plot the one-pagers and export? Default will take top
#' \code{robyn_clusters()} results.
#' @param csv_out Character. Accepts "pareto" or "all". Default to "pareto". Set
#' to "all" will output all iterations as csv. Set NULL to skip exports into CSVs.
#' @param ui Boolean. Save additional outputs for UI usage. List outcome.
#' @param export Boolean. Export outcomes into local files?
#' @param quiet Boolean. Keep messages off?
#' @param ... Additional parameters passed to \code{robyn_clusters()}
#' @return (Invisible) list with processed results.
#' @export
robyn_outputs <- function(InputCollect, OutputModels,
                          pareto_fronts = 1,
                          calibration_constraint = 0.1,
                          plot_folder = getwd(), plot_folder_sub = NULL,
                          plot_pareto = TRUE,
                          csv_out = "pareto",
                          clusters = TRUE, selected = "clusters",
                          ui = FALSE, export = TRUE,
                          quiet = FALSE, ...) {

  check_robyn_object(plot_folder)
  plot_folder <- check_filedir(plot_folder)

  # Check calibration constrains
  calibration_constraint <- check_calibconstr(
    calibration_constraint,
    InputCollect$iterations,
    InputCollect$trials,
    InputCollect$calibration_input)

  #####################################
  #### Run robyn_pareto on OutputModels

  totalModels <- InputCollect$iterations * InputCollect$trials
  message(sprintf(">>> Running Pareto calculations for %s models on %s front%s...",
                  totalModels, pareto_fronts, ifelse(pareto_fronts > 1, "s", "")))
  pareto_results <- robyn_pareto(InputCollect, OutputModels, pareto_fronts, calibration_constraint)
  allSolutions <- unique(pareto_results$xDecompVecCollect$solID)

  #####################################
  #### Gather the results into output object

  # Set folder to save outputs
  if (is.null(plot_folder_sub)) {
    refresh <- attr(OutputModels, "refresh")
    folder_var <- ifelse(!refresh, "init", paste0("rf", InputCollect$refreshCounter))
    plot_folder_sub <- paste0(format(Sys.time(), "%Y-%m-%d %H.%M"), " ", folder_var)
  }
  plotPath <- dir.create(file.path(plot_folder, plot_folder_sub))

  # Auxiliary list with all results (wasn't previously exported but needed for robyn_outputs())
  allPareto <- list(resultHypParam = pareto_results$resultHypParam,
                    xDecompAgg = pareto_results$xDecompAgg,
                    plotDataCollect = pareto_results$plotDataCollect)

  # Final results object
  OutputCollect <- list(
    resultHypParam = pareto_results$resultHypParam[solID %in% allSolutions],
    xDecompAgg = pareto_results$xDecompAgg[solID %in% allSolutions],
    mediaVecCollect = pareto_results$mediaVecCollect,
    xDecompVecCollect = pareto_results$xDecompVecCollect,
    OutputModels = OutputModels,
    allSolutions = allSolutions,
    allPareto = allPareto,
    calibration_constraint = calibration_constraint,
    UI = NULL,
    pareto_fronts = pareto_fronts,
    hyper_fixed = attr(OutputModels, "hyper_fixed"),
    plot_folder = paste0(plot_folder, "/", plot_folder_sub, "/")
  )

  class(OutputCollect) <- c("robyn_outputs", class(OutputCollect))

  if (export) {

    tryCatch({

      if (!quiet) message(paste0(">>> Collecting ", length(allSolutions)," pareto-optimum results into: ", OutputCollect$plot_folder))

      if (csv_out %in% c("all", "pareto")) {
        if (!quiet) message(paste(">> Exporting", csv_out, "results as CSVs into directory..."))
        robyn_csv(OutputCollect, csv_out, export = export)
      }

      if (!quiet) message(">> Exporting general plots into directory...")
      all_plots <- robyn_plots(InputCollect, OutputCollect, export = export)

      if (clusters) {
        if (!quiet) message(">>> Calculating clusters for model selection using Pareto fronts...")
        OutputCollect[["clusters"]] <- robyn_clusters(OutputCollect, quiet = quiet, export = export, ...)
      }

      if (plot_pareto) {
        if (!quiet) message(">>> Exporting pareto one-pagers into directory...")
        selected <- if (!clusters) NULL else selected
        pareto_onepagers <- robyn_onepagers(
          InputCollect, OutputCollect,
          selected = selected,
          quiet = quiet,
          export = export)
      }

      # For internal use -> UI Code
      if (ui & plot_pareto) OutputCollect$UI$pareto_onepagers <- pareto_onepagers
      OutputCollect[["UI"]] <- if (ui) list(pParFront = all_plots[["pParFront"]]) else NULL

    }, error = function(err) {
      message(paste("Failed exporting results, but returned model results anyways:\n", err))
    })
  }

  return(invisible(OutputCollect))

}


####################################################################
#' Output results into local files: CSV files
#'
#' @inheritParams robyn_outputs
#' @param OutputCollect \code{robyn_run(..., export = FALSE)} output
#' @export
robyn_csv <- function(OutputCollect, csv_out = NULL, export = TRUE) {
  if (export) {
    check_class("robyn_outputs", OutputCollect)
    temp_all <- OutputCollect$allPareto
    if ("pareto" %in% csv_out) {
      fwrite(OutputCollect$resultHypParam, paste0(OutputCollect$plot_folder, "pareto_hyperparameters.csv"))
      fwrite(OutputCollect$xDecompAgg, paste0(OutputCollect$plot_folder, "pareto_aggregated.csv"))
    }
    if ("all" %in% csv_out) {
      fwrite(temp_all$resultHypParam, paste0(OutputCollect$plot_folder, "all_hyperparameters.csv"))
      fwrite(temp_all$xDecompAgg, paste0(OutputCollect$plot_folder, "all_aggregated.csv"))
    }
    if (!is.null(csv_out)) {
      fwrite(OutputCollect$mediaVecCollect, paste0(OutputCollect$plot_folder, "pareto_media_transform_matrix.csv"))
      fwrite(OutputCollect$xDecompVecCollect, paste0(OutputCollect$plot_folder, "pareto_alldecomp_matrix.csv"))
    }
  }
}
