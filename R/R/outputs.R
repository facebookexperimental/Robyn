# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Output results into local files
#'
#' Pack \code{robyn_plots()}, \code{robyn_csv()}, and \code{robyn_clusters()}
#' outcomes for \code{robyn_run()} results. When \code{UI=TRUE}, enriched
#' \code{OutputCollect} results with additional plots and objects.
#'
#' @inheritParams robyn_run
#' @param InputCollect,OutputCollect \code{robyn_run()} outcomes.
#' @param selected Character vector. Which models (by \code{solID}) do you
#' wish to plot and export? Default will take top \code{robyn_clusters()} results.
#' @param quiet Boolean. Keep messages off?
#' @param ... Additional parameters passed to \code{robyn_clusters()}
#' @return (Invisible) enriched \code{OutputCollect}
#' @export
robyn_outputs <- function(InputCollect, OutputCollect,
                          csv_out = "all", clusters = TRUE,
                          selected = OutputCollect[["clusters"]]$models$solID,
                          plot_pareto = TRUE, ui = FALSE,
                          export = TRUE, quiet = FALSE, ...) {

  if (!quiet) message(">>> Collecting results:\n    For exported files, using directory: ", OutputCollect$plot_folder)

  if (csv_out %in% c("all", "pareto")) {
    if (!quiet) message(">> Exporting results as CSVs into directory...")
    robyn_csv(OutputCollect, csv_out, export = export)
  }

  if (!quiet) message(">> Exporting general plots into directory...")
  all_plots <- robyn_plots(InputCollect, OutputCollect, export = export)
  # For internal use -> UI Code
  OutputCollect[["UI"]] <- if (ui) list(pParFront = all_plots[["pParFront"]]) else NULL

  if (clusters) {
    if (!quiet) message(">>> Calculating clusters for model selection...")
    OutputCollect[["clusters"]] <- robyn_clusters(OutputCollect, quiet = quiet, export = export, ...)
  }

  if (plot_pareto) {
    if (!quiet) message(">> Exporting pareto one-pagers into directory...")
    pareto_onepagers <- robyn_onepagers(
      InputCollect, OutputCollect,
      selected = selected,
      quiet = quiet,
      export = export)
    # For internal use -> UI Code
    if (ui) OutputCollect$UI$pareto_onepagers <- pareto_onepagers
  }

  return(invisible(OutputCollect))

}


####################################################################
#' Output results into local files: CSV files
#'
#' Pack \code{robyn_pareto()}, \code{robyn_plots()}, \code{robyn_csv()},
#' and \code{robyn_cluster()} outcomes for \code{robyn_run()} results.
#'
#' @inheritParams robyn_outputs
#' @export
robyn_csv <- function(OutputCollect, csv_out = NULL, export = TRUE) {
  if (export) {
    check_class(OutputCollect, "robyn_run")
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
