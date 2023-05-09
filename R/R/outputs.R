# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Evaluate Models and Output Results into Local Files
#'
#' Pack \code{robyn_plots()}, \code{robyn_csv()}, and \code{robyn_clusters()}
#' outcomes on \code{robyn_run()} results. When \code{UI=TRUE}, enriched
#' \code{OutputModels} results with additional plots and objects.
#'
#' @param InputCollect,OutputModels \code{robyn_inputs()} and \code{robyn_run()}
#' outcomes.
#' @param pareto_fronts Integer. Number of Pareto fronts for the output.
#' \code{pareto_fronts = 1} returns the best models trading off \code{NRMSE} &
#' \code{DECOMP.RSSD}. Increase \code{pareto_fronts} to get more model choices.
#' \code{pareto_fronts = "auto"} selects the min fronts that include at least 100
#' candidates. To customize this threshold, set value with \code{min_candidates}.
#' @param calibration_constraint Numeric. Default to 0.1 and allows 0.01-0.1. When
#' calibrating, 0.1 means top 10% calibrated models are used for pareto-optimal
#' selection. Lower \code{calibration_constraint} increases calibration accuracy.
#' @param plot_folder Character. Path for saving plots. Default
#' to \code{robyn_object} and saves plot in the same directory as \code{robyn_object}.
#' @param plot_folder_sub Character. Sub path for saving plots. Will overwrite the
#' default path with timestamp.
#' @param plot_pareto Boolean. Set to \code{FALSE} to deactivate plotting
#' and saving model one-pagers. Used when testing models.
#' @param clusters Boolean. Apply \code{robyn_clusters()} to output models?
#' @param select_model Character vector. Which models (by \code{solID}) do you
#' wish to plot the one-pagers and export? Default will take top
#' \code{robyn_clusters()} results.
#' @param csv_out Character. Accepts "pareto" or "all". Default to "pareto". Set
#' to "all" will output all iterations as csv. Set NULL to skip exports into CSVs.
#' @param ui Boolean. Save additional outputs for UI usage. List outcome.
#' @param export Boolean. Export outcomes into local files?
#' @param all_sol_json Logical. Add all solutions to json export.
#' @param quiet Boolean. Keep messages off?
#' @param refresh Boolean. Refresh mode
#' @param ... Additional parameters passed to \code{robyn_clusters()}
#' @return (Invisible) list. Class: \code{robyn_outputs}. Contains processed
#' results based on \code{robyn_run()} results.
#' @export
robyn_outputs <- function(InputCollect, OutputModels,
                          pareto_fronts = "auto",
                          calibration_constraint = 0.1,
                          plot_folder = NULL,
                          plot_folder_sub = NULL,
                          plot_pareto = TRUE,
                          csv_out = "pareto",
                          clusters = TRUE,
                          select_model = "clusters",
                          ui = FALSE, export = TRUE,
                          all_sol_json = FALSE,
                          quiet = FALSE,
                          refresh = FALSE, ...) {
  if (is.null(plot_folder)) plot_folder <- getwd()
  plot_folder <- check_dir(plot_folder)

  # Check calibration constrains
  calibrated <- !is.null(InputCollect$calibration_input)
  all_fixed <- length(OutputModels$trial1$hyperBoundFixed) == length(OutputModels$hyper_updated)
  if (!all_fixed) {
    calibration_constraint <- check_calibconstr(
      calibration_constraint,
      OutputModels$iterations,
      OutputModels$trials,
      InputCollect$calibration_input,
      refresh = refresh
    )
  }

  #####################################
  #### Run robyn_pareto on OutputModels

  totalModels <- OutputModels$iterations * OutputModels$trials
  if (!isTRUE(attr(OutputModels, "hyper_fixed"))) {
    message(sprintf(
      ">>> Running Pareto calculations for %s models on %s front%s...",
      totalModels, pareto_fronts, ifelse(pareto_fronts > 1, "s", "")
    ))
  }
  pareto_results <- robyn_pareto(
    InputCollect, OutputModels,
    pareto_fronts = pareto_fronts,
    calibration_constraint = calibration_constraint,
    quiet = quiet,
    calibrated = calibrated,
    ...
  )
  pareto_fronts <- pareto_results$pareto_fronts
  allSolutions <- pareto_results$pareto_solutions

  #####################################
  #### Gather the results into output object

  # Auxiliary list with all results (wasn't previously exported but needed for robyn_outputs())
  allPareto <- list(
    resultHypParam = pareto_results$resultHypParam,
    xDecompAgg = pareto_results$xDecompAgg,
    resultCalibration = pareto_results$resultCalibration,
    plotDataCollect = pareto_results$plotDataCollect,
    df_caov_pct = pareto_results$df_caov_pct_all
  )

  # Set folder to save outputs: legacy plot_folder_sub
  depth <- ifelse(
    "refreshDepth" %in% names(InputCollect),
    InputCollect$refreshDepth,
    ifelse("refreshCounter" %in% names(InputCollect),
      InputCollect$refreshCounter, 0
    )
  )
  folder_var <- ifelse(!as.integer(depth) > 0, "init", paste0("rf", depth))
  if (is.null(plot_folder_sub)) {
    plot_folder_sub <- paste("Robyn", format(Sys.time(), "%Y%m%d%H%M"), folder_var, sep = "_")
  }

  # Final results object
  OutputCollect <- list(
    resultHypParam = filter(pareto_results$resultHypParam, .data$solID %in% allSolutions),
    xDecompAgg = filter(pareto_results$xDecompAgg, .data$solID %in% allSolutions),
    mediaVecCollect = pareto_results$mediaVecCollect,
    xDecompVecCollect = pareto_results$xDecompVecCollect,
    resultCalibration = if (calibrated) {
      filter(pareto_results$resultCalibration, .data$solID %in% allSolutions)
    } else {
      NULL
    },
    allSolutions = allSolutions,
    allPareto = allPareto,
    calibration_constraint = calibration_constraint,
    OutputModels = OutputModels,
    cores = OutputModels$cores,
    iterations = OutputModels$iterations,
    trials = OutputModels$trials,
    intercept_sign = OutputModels$intercept_sign,
    nevergrad_algo = OutputModels$nevergrad_algo,
    add_penalty_factor = OutputModels$add_penalty_factor,
    seed = OutputModels$seed,
    UI = NULL,
    pareto_fronts = pareto_fronts,
    hyper_fixed = attr(OutputModels, "hyper_fixed"),
    plot_folder = gsub("//", "", paste0(plot_folder, "/", plot_folder_sub, "/"))
  )
  class(OutputCollect) <- c("robyn_outputs", class(OutputCollect))

  plotPath <- paste0(plot_folder, "/", plot_folder_sub, "/")
  OutputCollect$plot_folder <- gsub("//", "/", plotPath)
  if (export && !dir.exists(OutputCollect$plot_folder)) dir.create(OutputCollect$plot_folder, recursive = TRUE)

  # Cluster results and amend cluster output
  if (clusters) {
    if (!quiet) message(">>> Calculating clusters for model selection using Pareto fronts...")
    try(clusterCollect <- robyn_clusters(OutputCollect,
      dep_var_type = InputCollect$dep_var_type,
      quiet = quiet, export = export, ...
    ))
    OutputCollect$resultHypParam <- left_join(
      OutputCollect$resultHypParam,
      select(clusterCollect$data, .data$solID, .data$cluster, .data$top_sol),
      by = "solID"
    )
    OutputCollect$xDecompAgg <- left_join(
      OutputCollect$xDecompAgg,
      select(clusterCollect$data, .data$solID, .data$cluster, .data$top_sol),
      by = "solID"
    ) %>%
      left_join(
        select(
          clusterCollect$df_cluster_ci, .data$rn, .data$cluster, .data$boot_mean,
          .data$boot_se, .data$ci_low, .data$ci_up, .data$rn
        ),
        by = c("rn", "cluster")
      ) %>%
      left_join(
        pareto_results$df_caov_pct_all,
        by = c("solID", "rn")
      )
    OutputCollect$mediaVecCollect <- left_join(
      OutputCollect$mediaVecCollect,
      select(clusterCollect$data, .data$solID, .data$cluster, .data$top_sol),
      by = "solID"
    )
    OutputCollect$xDecompVecCollect <- left_join(
      OutputCollect$xDecompVecCollect,
      select(clusterCollect$data, .data$solID, .data$cluster, .data$top_sol),
      by = "solID"
    )
    if (calibrated) {
      OutputCollect$resultCalibration <- left_join(
        OutputCollect$resultCalibration,
        select(clusterCollect$data, .data$solID, .data$cluster, .data$top_sol),
        by = "solID"
      )
    }
    OutputCollect[["clusters"]] <- clusterCollect
  }

  if (export) {
    tryCatch(
      {
        if (!quiet) message(paste0(">>> Collecting ", length(allSolutions), " pareto-optimum results into: ", OutputCollect$plot_folder))

        if (!quiet) message(">> Exporting general plots into directory...")
        all_plots <- robyn_plots(InputCollect, OutputCollect, export = export)

        if (csv_out %in% c("all", "pareto")) {
          if (!quiet) message(paste(">> Exporting", csv_out, "results as CSVs into directory..."))
          robyn_csv(InputCollect, OutputCollect, csv_out, export = export, calibrated = calibrated)
        }

        if (plot_pareto) {
          if (!quiet) {
            message(sprintf(
              ">>> Exporting %sone-pagers into directory...", ifelse(!OutputCollect$hyper_fixed, "pareto ", "")
            ))
          }
          select_model <- if (!clusters || is.null(OutputCollect[["clusters"]])) NULL else select_model
          pareto_onepagers <- robyn_onepagers(
            InputCollect, OutputCollect,
            select_model = select_model,
            quiet = quiet,
            export = export
          )
        }

        if (all_sol_json) {
          all_sol_json <- OutputCollect$resultHypParam %>%
            filter(!is.na(.data$cluster)) %>%
            select(c("solID", "cluster", "top_sol")) %>%
            arrange(.data$cluster, -.data$top_sol, .data$solID)
        } else {
          all_sol_json <- NULL
        }
        robyn_write(
          InputCollect = InputCollect, OutputModels = OutputModels,
          dir = OutputCollect$plot_folder, quiet = quiet, all_sol_json = all_sol_json
        )

        # For internal use -> UI Code
        if (ui && plot_pareto) OutputCollect$UI$pareto_onepagers <- pareto_onepagers
        OutputCollect[["UI"]] <- if (ui) list(pParFront = all_plots[["pParFront"]]) else NULL
      },
      error = function(err) {
        message(paste("Failed exporting results, but returned model results anyways:\n", err))
      }
    )
  }

  if (!is.null(OutputModels$hyper_updated)) OutputCollect$hyper_updated <- OutputModels$hyper_updated
  class(OutputCollect) <- c("robyn_outputs", class(OutputCollect))
  return(invisible(OutputCollect))
}

#' @rdname robyn_outputs
#' @aliases robyn_outputs
#' @param x \code{robyn_outputs()} output.
#' @export
print.robyn_outputs <- function(x, ...) {
  print(glued(
    "
Plot Folder: {x$plot_folder}
Calibration Constraint: {x$calibration_constraint}
Hyper-parameters fixed: {x$hyper_fixed}
Pareto-front ({x$pareto_fronts}) All solutions ({nSols}): {paste(x$allSolutions, collapse = ', ')}
{clusters_info}
",
    nSols = length(x$allSolutions),
    clusters_info = if ("clusters" %in% names(x)) {
      glued(
        "Clusters (k = {x$clusters$n_clusters}): {paste(x$clusters$models$solID, collapse = ', ')}"
      )
    } else {
      NULL
    }
  ))
}


####################################################################
#' Output results into local files: CSV files
#'
#' @param OutputCollect \code{robyn_run(..., export = FALSE)} output.
#' @param calibrated Logical
#' @rdname robyn_outputs
#' @return Invisible \code{NULL}.
#' @export
robyn_csv <- function(InputCollect, OutputCollect, csv_out = NULL, export = TRUE, calibrated = FALSE) {
  if (export) {
    check_class("robyn_outputs", OutputCollect)
    temp_all <- OutputCollect$allPareto
    if ("pareto" %in% csv_out) {
      write.csv(OutputCollect$resultHypParam, paste0(OutputCollect$plot_folder, "pareto_hyperparameters.csv"))
      write.csv(OutputCollect$xDecompAgg, paste0(OutputCollect$plot_folder, "pareto_aggregated.csv"))
      if (calibrated) {
        write.csv(OutputCollect$resultCalibration, paste0(OutputCollect$plot_folder, "pareto_calibration.csv"))
      }
    }
    if ("all" %in% csv_out) {
      write.csv(temp_all$resultHypParam, paste0(OutputCollect$plot_folder, "all_hyperparameters.csv"))
      write.csv(temp_all$xDecompAgg, paste0(OutputCollect$plot_folder, "all_aggregated.csv"))
      if (calibrated) {
        write.csv(temp_all$resultCalibration, paste0(OutputCollect$plot_folder, "all_calibration.csv"))
      }
    }
    if (!is.null(csv_out)) {
      write.csv(InputCollect$dt_input, paste0(OutputCollect$plot_folder, "raw_data.csv"))
      write.csv(OutputCollect$mediaVecCollect, paste0(OutputCollect$plot_folder, "pareto_media_transform_matrix.csv"))
      write.csv(OutputCollect$xDecompVecCollect, paste0(OutputCollect$plot_folder, "pareto_alldecomp_matrix.csv"))
    }
  }
}
