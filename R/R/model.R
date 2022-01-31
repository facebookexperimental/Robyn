# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Includes function robyn_run, robyn_mmm, model_refit, model_decomp, calibrate_mmm, ridge_lambda

####################################################################
#' The major Robyn modelling function
#'
#' The \code{robyn_run()} function consumes output from \code{robyn_input()},
#' runs the \code{robyn_mmm()} functions and plots and collects the result.
#'
#' @inheritParams robyn_allocator
#' @param plot_folder Character. Path for saving plots. Default
#' to \code{robyn_object} and saves plot in the same directory as \code{robyn_object}.
#' @param plot_folder_sub Character. Customize sub path to save plots. The total
#' path is created with \code{dir.create(file.path(plot_folder, plot_folder_sub))}.
#' For example, plot_folder_sub = "sub_dir".
#' @param dt_hyper_fixed data.frame. Only provide when loading old model results.
#' It consumes hyperparameters from saved csv \code{pareto_hyperparameters.csv}.
#' @param pareto_fronts Integer. Number of Pareto fronts for the output.
#' \code{pareto_fronts = 1} returns the best models trading off \code{NRMSE} &
#' \code{DECOMP.RSSD}. Increase \code{pareto_fronts} to get more model choices.
#' @param plot_pareto Boolean. Set to \code{FALSE} to deactivate plotting
#' and saving model one-pagers. Used when testing models.
#' @param clusters Boolean. Apply \code{robyn_clusters()} to output models?
#' @param calibration_constraint Numeric. Default to 0.1 and allows 0.01-0.1. When
#' calibrating, 0.1 means top 10% calibrated models are used for pareto-optimal
#' selection. Lower \code{calibration_constraint} increases calibration accuracy.
#' @param lambda_control Numeric. From 0-1. Tunes ridge lambda between
#' lambda.min and lambda.1se.
#' @param refresh Boolean. Set to \code{TRUE} when used in \code{robyn_refresh()}.
#' @param seed Integer. For reproducible results when running nevergrad.
#' @param csv_out Character. Accepts "pareto" or "all". Default to "pareto". Set
#' to "all" will output all iterations as csv. Set NULL to skip exports into CSVs.
#' @param ui Boolean. Save additional outputs for UI usage. List outcome.
#' @param export Boolean. Export results into local files.
#' @param ... Additional parameters passed to \code{robyn_clusters()}.
#' @examples
#' \dontrun{
#' OutputCollect <- robyn_run(
#'   InputCollect = InputCollect,
#'   plot_folder = robyn_object,
#'   pareto_fronts = 3,
#'   plot_pareto = TRUE
#' )
#' }
#' @export
robyn_run <- function(InputCollect,
                      plot_folder = getwd(),
                      plot_folder_sub = NULL,
                      pareto_fronts = 1,
                      plot_pareto = TRUE,
                      clusters = TRUE,
                      calibration_constraint = 0.1,
                      lambda_control = 1,
                      refresh = FALSE,
                      dt_hyper_fixed = NULL,
                      seed = 123L,
                      csv_out = "pareto",
                      ui = FALSE,
                      export = TRUE,
                      ...) {

  t0 <- Sys.time()

  #####################################
  #### Set local environment

  if (!"hyperparameters" %in% names(InputCollect)) {
    stop("Must provide 'hyperparameters' in robyn_inputs()'s output first")
  }

  init_msgs_run(InputCollect, refresh)

  # Run some checks
  check_robyn_object(plot_folder)
  plot_folder <- check_filedir(plot_folder)
  calibration_constraint <- check_calibconstr(
    calibration_constraint, InputCollect$iterations,
    InputCollect$trials, InputCollect$calibration_input)

  #####################################
  #### Run robyn_mmm on set_trials

  hyper_fixed <- check_hyper_fixed(InputCollect, dt_hyper_fixed)
  model_output_collect <- robyn_train(InputCollect, lambda_control, refresh, seed, dt_hyper_fixed)

  #####################################
  #### Run robyn_pareto on model_output_collect

  message(">>> Running pareto calculations...")
  pareto_results <- robyn_pareto(InputCollect, model_output_collect, pareto_fronts, calibration_constraint, hyper_fixed)
  allSolutions <- unique(pareto_results$xDecompVecCollect$solID)

  #####################################
  #### Gather the results into output object

  # Set folder to save outputs
  if (is.null(plot_folder_sub)) {
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
    model_output_collect = model_output_collect,
    allSolutions = allSolutions,
    allPareto = allPareto,
    UI = NULL,
    pareto_fronts = pareto_fronts,
    hyper_fixed = hyper_fixed,
    plot_folder = paste0(plot_folder, "/", plot_folder_sub, "/")
  )
  class(OutputCollect) <- c("robyn_run", class(OutputCollect))

  #####################################
  #### Collect and export results

  if (export) {
    tryCatch({
      OutputCollect <- robyn_outputs(
        InputCollect, OutputCollect,
        csv_out = csv_out, clusters = clusters,
        plot_pareto = plot_pareto, ui = ui, ...)
    }, error = function(err) {
      message(paste("+ Failed exporting results, but returned model results anyways:\n", err))
    })
  }

  # Report total timing
  OutputCollect[["totalTime"]] <- round(difftime(Sys.time(), t0, units = "mins"), 2)
  message(paste("\nTotal time:", OutputCollect[["totalTime"]], "mins"))

  return(invisible(OutputCollect))

}


####################################################################
#' Train Robyn Models
#'
#' The \code{robyn_train()} function consumes output from \code{robyn_input()}
#' and runs the \code{robyn_mmm()} on each trial.
#'
#' @inheritParams robyn_run
#' @examples
#' \dontrun{
#' OutputCollect <- robyn_train(
#'   InputCollect = InputCollect,
#'   dt_hyper_fixed = NULL,
#'   seed = 0
#' )
#' }
#' @export
robyn_train <- function(InputCollect, lambda_control = 1, refresh = FALSE, seed = 123, dt_hyper_fixed = NULL) {

  hyper_fixed <- check_hyper_fixed(InputCollect, dt_hyper_fixed)

  if (!is.null(dt_hyper_fixed)) {

    ## Run robyn_mmm if using old model result tables

    dt_hyper_fixed <- as.data.table(dt_hyper_fixed)
    hypParamSamName <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
    hyperparameters_fixed <- lapply(dt_hyper_fixed[, hypParamSamName, with = FALSE], unlist)

    model_output_collect <- list()
    model_output_collect[[1]] <- robyn_mmm(
      hyper_collect = hyperparameters_fixed,
      InputCollect = InputCollect,
      lambda_fixed = dt_hyper_fixed$lambda,
      seed = seed
    )

    model_output_collect[[1]]$trial <- 1
    model_output_collect[[1]]$resultCollect$resultHypParam <- model_output_collect[[1]]$resultCollect$resultHypParam[order(iterPar)]
    dt_IDs <- data.table(
      solID = dt_hyper_fixed$solID,
      iterPar = model_output_collect[[1]]$resultCollect$resultHypParam$iterPar
    )
    model_output_collect[[1]]$resultCollect$resultHypParam[dt_IDs, on = .(iterPar), "solID" := .(i.solID)]
    model_output_collect[[1]]$resultCollect$xDecompAgg[dt_IDs, on = .(iterPar), "solID" := .(i.solID)]
    model_output_collect[[1]]$resultCollect$xDecompVec[dt_IDs, on = .(iterPar), "solID" := .(i.solID)]
    model_output_collect[[1]]$resultCollect$decompSpendDist[dt_IDs, on = .(iterPar), "solID" := .(i.solID)]

  } else {

    ## Run robyn_mmm on set_trials if hyperparameters are not all fixed

    check_parallel_msg(InputCollect)

    message(paste(
      ">>> Start running", InputCollect$trials, "trials with",
      InputCollect$iterations, "iterations per trial each",
      ifelse(is.null(InputCollect$calibration_input), "with", "with calibration and"),
      InputCollect$nevergrad_algo, "nevergrad algorithm..."
    ))

    model_output_collect <- list()

    for (ngt in 1:InputCollect$trials) {
      message(paste0("  Running trial #", ngt))
      model_output <- robyn_mmm(
        hyper_collect = InputCollect$hyperparameters,
        InputCollect = InputCollect,
        lambda_control = lambda_control,
        refresh = refresh,
        seed = seed
      )
      check_coef0 <- any(model_output$resultCollect$decompSpendDist$decomp.rssd == Inf)
      if (check_coef0) {
        num_coef0_mod <- model_output$resultCollect$decompSpendDist[decomp.rssd == Inf, uniqueN(paste0(iterNG, "_", iterPar))]
        num_coef0_mod <- ifelse(num_coef0_mod > InputCollect$iterations, InputCollect$iterations, num_coef0_mod)
        message("This trial contains ", num_coef0_mod, " iterations with all 0 media coefficient. Please reconsider your media variable choice if the pareto choices are unreasonable.
                  \nRecommendations are: \n1. increase hyperparameter ranges for 0-coef channels to give Robyn more freedom\n2. split media into sub-channels, and/or aggregate similar channels, and/or introduce other media\n3. increase trials to get more samples\n")
      }
      model_output["trial"] <- ngt
      model_output_collect[[ngt]] <- model_output
    }
  }
  names(model_output_collect) <- if (hyper_fixed) "trial1" else paste0("trial", 1:InputCollect$trials)
  return(model_output_collect)
}

init_msgs_run <- function(InputCollect, refresh) {
  message(sprintf(
    "Input data has %s %ss in total: %s to %s",
    nrow(InputCollect$dt_mod),
    InputCollect$intervalType,
    min(InputCollect$dt_mod$ds),
    max(InputCollect$dt_mod$ds)
  ))
  message(sprintf(
    "%s model is built on rolling window of %s %s: %s to %s",
    ifelse(!refresh, "Initial", "Refresh"),
    InputCollect$rollingWindowLength,
    InputCollect$intervalType,
    InputCollect$window_start,
    InputCollect$window_end
  ))
  if (refresh) {
    message("Rolling window moving forward: ", InputCollect$refresh_steps, " ", InputCollect$intervalType)
  }
}

####################################################################
#' The core MMM function
#'
#' The \code{robyn_mmm()} function activates Nevergrad to generate samples of
#' hyperparameters, conducts media transformation within each loop, fits the
#' Ridge regression, calibrates the model optionally, decomposes responses
#' and collects the result. It's an inner function within \code{robyn_run()}.
#'
#' @inheritParams robyn_run
#' @inheritParams robyn_allocator
#' @param hyper_collect List. Containing hyperparameter bounds. Defaults to
#' \code{InputCollect$hyperparameters}.
#' @param iterations Integer. Number of iterations to run.
#' @param lambda.n Integer. Number of lambda cross-validation in \code{glmnet}.
#' Defaults to 100.
#' @param lambda_fixed Boolean. \code{lambda_fixed = TRUE} when inputting
#' old model results.
#' @export
robyn_mmm <- function(hyper_collect,
                      InputCollect,
                      iterations = InputCollect$iterations,
                      lambda.n = 100,
                      lambda_control = 1,
                      lambda_fixed = NULL,
                      refresh = FALSE,
                      seed = 123L) {
  if (reticulate::py_module_available("nevergrad")) {
    ng <- reticulate::import("nevergrad", delay_load = TRUE)
    if (is.integer(seed)) {
      np <- reticulate::import("numpy", delay_load = FALSE)
      np$random$seed(seed)
    }
  } else {
    stop("You must have nevergrad python library installed.")
  }

  ################################################
  #### Collect hyperparameters

  hypParamSamName <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
  hyper_fixed <- FALSE

  # hyper_collect <- unlist(list(...), recursive = FALSE) # hyper_collect <- InputCollect$hyperparameters; hyper_collect <- hyperparameters_fixed

  # sort hyperparameter list by name
  hyper_bound_list <- list()
  for (i in 1:length(hypParamSamName)) {
    hyper_bound_list[i] <- hyper_collect[hypParamSamName[i]]
    names(hyper_bound_list)[i] <- hypParamSamName[i]
  }

  # get hyperparameters for Nevergrad
  hyper_which <- which(sapply(hyper_bound_list, length) == 2)
  hyper_bound_list_updated <- hyper_bound_list[hyper_which]
  hyper_bound_list_updated_name <- names(hyper_bound_list_updated)
  hyper_count <- length(hyper_bound_list_updated)
  if (hyper_count == 0) {
    hyper_fixed <- TRUE
    if (is.null(lambda_fixed)) {
      stop("when hyperparameters are fixed, lambda_fixed must be provided from the selected lambda in old model")
    }
  }

  # get fixed hyperparameters
  hyper_fixed_which <- which(sapply(hyper_bound_list, length) == 1)
  hyper_bound_list_fixed <- hyper_bound_list[hyper_fixed_which]
  hyper_bound_list_fixed_name <- names(hyper_bound_list_fixed)
  hyper_count_fixed <- length(hyper_bound_list_fixed)

  # hyper_bound_list_fixed <- list(print_S_alphas = 1 , print_S_gammas = 0.5)
  if (InputCollect$cores > 1) {
    dt_hyperFixed <- data.table(sapply(hyper_bound_list_fixed, function(x) rep(x, InputCollect$cores)))
  } else {
    dt_hyperFixed <- as.data.table(matrix(hyper_bound_list_fixed, nrow = 1))
    names(dt_hyperFixed) <- hyper_bound_list_fixed_name
  }

  ################################################
  #### Setup environment

  if (is.null(InputCollect$dt_mod)) {
    stop("Run InputCollect$dt_mod <- robyn_engineering() first to get the dt_mod")
  }

  ## get environment for parallel backend
  InputCollect <- InputCollect
  dt_mod <- copy(InputCollect$dt_mod)
  xDecompAggPrev <- InputCollect$xDecompAggPrev
  rollingWindowStartWhich <- InputCollect$rollingWindowStartWhich
  rollingWindowEndWhich <- InputCollect$rollingWindowEndWhich
  refreshAddedStart <- InputCollect$refreshAddedStart
  dt_modRollWind <- copy(InputCollect$dt_modRollWind)
  refresh_steps <- InputCollect$refresh_steps
  rollingWindowLength <- InputCollect$rollingWindowLength

  paid_media_vars <- InputCollect$paid_media_vars
  paid_media_spends <- InputCollect$paid_media_spends
  organic_vars <- InputCollect$organic_vars
  context_vars <- InputCollect$context_vars
  prophet_vars <- InputCollect$prophet_vars
  adstock <- InputCollect$adstock
  context_signs <- InputCollect$context_signs
  paid_media_signs <- InputCollect$paid_media_signs
  prophet_signs <- InputCollect$prophet_signs
  organic_signs <- InputCollect$organic_signs
  all_media <- InputCollect$all_media
  # factor_vars <- InputCollect$factor_vars
  calibration_input <- InputCollect$calibration_input
  optimizer_name <- InputCollect$nevergrad_algo
  cores <- InputCollect$cores

  ################################################
  #### Get spend share

  dt_inputTrain <- InputCollect$dt_input[rollingWindowStartWhich:rollingWindowEndWhich]
  dt_spendShare <- dt_inputTrain[, .(
    rn = paid_media_vars,
    total_spend = sapply(.SD, sum),
    mean_spend = sapply(.SD, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0])))
  ), .SDcols = paid_media_spends]
  dt_spendShare[, ":="(spend_share = total_spend / sum(total_spend))]

  refreshAddedStartWhich <- which(dt_modRollWind$ds == refreshAddedStart)
  dt_spendShareRF <- dt_inputTrain[
    refreshAddedStartWhich:rollingWindowLength,
    .(rn = paid_media_vars,
      total_spend = sapply(.SD, sum),
      mean_spend = sapply(.SD, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0])))
    ),
    .SDcols = paid_media_spends
  ]
  dt_spendShareRF[, ":="(spend_share = total_spend / sum(total_spend))]
  dt_spendShare[, ":="(total_spend_refresh = dt_spendShareRF$total_spend,
                       mean_spend_refresh = dt_spendShareRF$mean_spend,
                       spend_share_refresh = dt_spendShareRF$spend_share)]


  ################################################
  #### Start Nevergrad loop

  t0 <- Sys.time()

  ## set iterations

  if (hyper_fixed == FALSE) {
    iterTotal <- iterations
    iterPar <- cores
  } else {
    iterTotal <- 1
    iterPar <- 1
  }

  iterNG <- ifelse(hyper_fixed == FALSE, ceiling(iterations / cores), 1)

  # cat("\nRunning", iterTotal,"iterations with evolutionary algorithm on",adstock, "adstocking,", length(hyper_bound_list_updated),"hyperparameters,",lambda.n,"-fold ridge x-validation using", cores,"cores...\n")

  ## start Nevergrad optimiser

  if (length(hyper_bound_list_updated) != 0) {
    my_tuple <- tuple(hyper_count)
    instrumentation <- ng$p$Array(shape = my_tuple, lower = 0., upper = 1.)
    # instrumentation$set_bounds(0., 1.)
    optimizer <- ng$optimizers$registry[optimizer_name](instrumentation, budget = iterTotal, num_workers = cores)
    if (is.null(calibration_input)) {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1.0, 1.0))
    } else {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1.0, 1.0, 1.0))
    }
    # Creating a hyperparameter vector to be used in the next learning.
  }

  ## start loop

  resultCollectNG <- list()
  cnt <- 0
  if (hyper_fixed == FALSE) {
    pb <- txtProgressBar(max = iterTotal, style = 3)
  }
  # assign("InputCollect", InputCollect, envir = .GlobalEnv) # adding this to enable InputCollect reading during parallel
  # opts <- list(progress = function(n) setTxtProgressBar(pb, n))

  # enable parallelisation of main modelling loop for MacOS and Linux only
  # create cluster before big for-loop to minimize overhead for parallel backend registering
  if (check_parallel()) {
    registerDoParallel(InputCollect$cores)
  } else {
    registerDoSEQ()
  }

  sysTimeDopar <- system.time({
    for (lng in 1:iterNG) { # lng = 1
      nevergrad_hp <- list()
      nevergrad_hp_val <- list()
      hypParamSamList <- list()
      hypParamSamNG <- c()

      if (hyper_fixed == FALSE) {
        for (co in 1:iterPar) { # co = 1

          ## get hyperparameter sample with ask
          nevergrad_hp[[co]] <- optimizer$ask()
          nevergrad_hp_val[[co]] <- nevergrad_hp[[co]]$value

          ## scale sample to given bounds
          for (hypNameLoop in hyper_bound_list_updated_name) { # hypNameLoop <- local_name.all[1]
            index <- which(hypNameLoop == hyper_bound_list_updated_name)
            channelBound <- unlist(hyper_bound_list_updated[hypNameLoop])
            hyppar_for_qunif <- nevergrad_hp_val[[co]][index]
            hyppar_scaled <- qunif(hyppar_for_qunif, min(channelBound), max(channelBound))
            hypParamSamNG[hypNameLoop] <- hyppar_scaled
          }
          hypParamSamList[[co]] <- transpose(data.table(hypParamSamNG))
        }

        hypParamSamNG <- rbindlist(hypParamSamList)
        hypParamSamNG <- setnames(hypParamSamNG, names(hypParamSamNG), hyper_bound_list_updated_name)

        ## add fixed hyperparameters

        if (hyper_count_fixed != 0) {
          hypParamSamNG <- cbind(hypParamSamNG, dt_hyperFixed)
          hypParamSamNG <- setcolorder(hypParamSamNG, hypParamSamName)
        }
      } else {
        hypParamSamNG <- as.data.table(matrix(unlist(hyper_bound_list), nrow = 1))
        setnames(hypParamSamNG, names(hypParamSamNG), hypParamSamName)
      }

      ## Parallel start

      nrmse.collect <- c()
      decomp.rssd.collect <- c()
      best_mape <- Inf

      doparCollect <- suppressPackageStartupMessages(
        foreach(i = 1:iterPar) %dorng% { # i = 1
          t1 <- Sys.time()

          #####################################
          #### Get hyperparameter sample

          hypParamSam <- unlist(hypParamSamNG[i])

          #### Tranform media with hyperparameters
          dt_modAdstocked <- dt_mod[, .SD, .SDcols = setdiff(names(dt_mod), "ds")]
          mediaAdstocked <- list()
          mediaVecCum <- list()
          mediaSaturated <- list()
          for (v in 1:length(all_media)) {
            m <- dt_modAdstocked[, get(all_media[v])]

            ## adstocking

            if (adstock == "geometric") {
              theta <- hypParamSam[paste0(all_media[v], "_thetas")]
              x_list <- adstock_geometric(x = m, theta = theta)
            } else if (adstock == "weibull_cdf") {
              shape <- hypParamSam[paste0(all_media[v], "_shapes")]
              scale <- hypParamSam[paste0(all_media[v], "_scales")]
              x_list <- adstock_weibull(x = m, shape = shape, scale = scale, windlen = rollingWindowLength, type = "cdf")
            } else if (adstock == "weibull_pdf") {
              shape <- hypParamSam[paste0(all_media[v], "_shapes")]
              scale <- hypParamSam[paste0(all_media[v], "_scales")]
              x_list <- adstock_weibull(x = m, shape = shape, scale = scale, windlen = rollingWindowLength, type = "pdf")
            } else {
              break
              print("adstock parameter must be geometric, weibull_cdf or weibull_pdf")
            }

            m_adstocked <- x_list$x_decayed
            mediaAdstocked[[v]] <- m_adstocked
            mediaVecCum[[v]] <- x_list$thetaVecCum

            ## saturation
            m_adstockedRollWind <- m_adstocked[rollingWindowStartWhich:rollingWindowEndWhich]

            alpha <- hypParamSam[paste0(all_media[v], "_alphas")]
            gamma <- hypParamSam[paste0(all_media[v], "_gammas")]
            mediaSaturated[[v]] <- saturation_hill(m_adstockedRollWind, alpha = alpha, gamma = gamma)
          }

          names(mediaAdstocked) <- all_media
          dt_modAdstocked[, (all_media) := mediaAdstocked]
          dt_mediaVecCum <- data.table()[, (all_media) := mediaVecCum]

          names(mediaSaturated) <- all_media
          dt_modSaturated <- dt_modAdstocked[rollingWindowStartWhich:rollingWindowEndWhich]
          dt_modSaturated[, (all_media) := mediaSaturated]

          #####################################
          #### Split and prepare data for modelling

          dt_train <- copy(dt_modSaturated)

          ## contrast matrix because glmnet does not treat categorical variables
          y_train <- dt_train$dep_var
          x_train <- model.matrix(dep_var ~ ., dt_train)[, -1]

          ## create lambda sequence with x and y
          # lambda_seq <- ridge_lambda(x=x_train, y=y_train, seq_len = lambda.n, lambda_min_ratio = 0.0001)

          ## define sign control
          dt_sign <- dt_modSaturated[, !"dep_var"] # names(dt_sign)
          x_sign <- c(prophet_signs, context_signs, paid_media_signs, organic_signs)
          names(x_sign) <- c(prophet_vars, context_vars, paid_media_vars, organic_vars)
          check_factor <- sapply(dt_sign, is.factor)

          lower.limits <- c()
          upper.limits <- c()

          for (s in 1:length(check_factor)) {
            if (check_factor[s] == TRUE) {
              level.n <- length(levels(unlist(dt_sign[, s, with = FALSE])))
              if (level.n <= 1) {
                stop("factor variables must have more than 1 level")
              }
              lower_vec <- if (x_sign[s] == "positive") {
                rep(0, level.n - 1)
              } else {
                rep(-Inf, level.n - 1)
              }
              upper_vec <- if (x_sign[s] == "negative") {
                rep(0, level.n - 1)
              } else {
                rep(Inf, level.n - 1)
              }
              lower.limits <- c(lower.limits, lower_vec)
              upper.limits <- c(upper.limits, upper_vec)
            } else {
              lower.limits <- c(lower.limits, ifelse(x_sign[s] == "positive", 0, -Inf))
              upper.limits <- c(upper.limits, ifelse(x_sign[s] == "negative", 0, Inf))
            }
          }

          #####################################
          #### fit ridge regression with x-validation
          cvmod <- cv.glmnet(
            x_train,
            y_train,
            family = "gaussian",
            alpha = 0 # 0 for ridge regression
            # ,lambda = lambda_seq
            , lower.limits = lower.limits,
            upper.limits = upper.limits,
            type.measure = "mse"
            # ,penalty.factor = c(1,1,1,1,1,1,1,1,1)
            # ,nlambda = 100
            # ,nfold = 10
            # ,intercept = FALSE
          ) # plot(cvmod) coef(cvmod)
          # head(predict(cvmod, newx=x_train, s="lambda.1se"))

          lambda_range <- c(cvmod$lambda.min, cvmod$lambda.1se)
          lambda <- lambda_range[1] + (lambda_range[2]-lambda_range[1]) * lambda_control

          #####################################
          #### refit ridge regression with selected lambda from x-validation

          ## if no lift calibration, refit using best lambda
          if (hyper_fixed == FALSE) {
            mod_out <- model_refit(x_train, y_train, lambda = lambda, lower.limits, upper.limits)
          } else {
            mod_out <- model_refit(x_train, y_train, lambda = lambda_fixed[i], lower.limits, upper.limits)
            lambda <- lambda_fixed[i]
          }

          # hypParamSam["lambdas"] <- cvmod$lambda.1se
          # hypParamSamName <- names(hypParamSam)

          decompCollect <- model_decomp(coefs = mod_out$coefs, dt_modSaturated = dt_modSaturated, x = x_train, y_pred = mod_out$y_pred, i = i, dt_modRollWind = dt_modRollWind, refreshAddedStart = refreshAddedStart)
          nrmse <- mod_out$nrmse_train
          mape <- 0
          df.int <- mod_out$df.int


          #####################################
          #### get calibration mape

          if (!is.null(calibration_input)) {
            liftCollect <- calibrate_mmm(decompCollect = decompCollect, calibration_input = calibration_input, paid_media_vars = paid_media_vars, dayInterval = InputCollect$dayInterval)
            mape <- liftCollect[, mean(mape_lift)]
          }

          #####################################
          #### calculate multi-objectives for pareto optimality

          ## decomp objective: sum of squared distance between decomp share and spend share to be minimised
          dt_decompSpendDist <- decompCollect$xDecompAgg[rn %in% paid_media_vars, .(rn, xDecompAgg, xDecompPerc, xDecompMeanNon0Perc, xDecompMeanNon0, xDecompPercRF, xDecompMeanNon0PercRF, xDecompMeanNon0RF)]
          dt_decompSpendDist <- dt_decompSpendDist[dt_spendShare[, .(rn, spend_share, spend_share_refresh, mean_spend, total_spend)], on = "rn"]
          dt_decompSpendDist[, ":="(effect_share = xDecompPerc / sum(xDecompPerc),
                                    effect_share_refresh = xDecompPercRF / sum(xDecompPercRF))]
          decompCollect$xDecompAgg[dt_decompSpendDist[, .(rn, spend_share_refresh, effect_share_refresh)],
                                   ":="(spend_share_refresh = i.spend_share_refresh,
                                        effect_share_refresh = i.effect_share_refresh),
                                   on = "rn"
          ]

          if (!refresh) {
            decomp.rssd <- dt_decompSpendDist[, sqrt(sum((effect_share - spend_share)^2))]
          } else {
            dt_decompRF <- decompCollect$xDecompAgg[, .(rn, decomp_perc = xDecompPerc)][xDecompAggPrev[, .(rn, decomp_perc_prev = xDecompPerc)], on = "rn"]
            decomp.rssd.nonmedia <- dt_decompRF[!(rn %in% paid_media_vars), sqrt(mean((decomp_perc - decomp_perc_prev)^2))]
            decomp.rssd.media <- dt_decompSpendDist[, sqrt(mean((effect_share_refresh - spend_share_refresh)^2))]
            decomp.rssd <- decomp.rssd.media + decomp.rssd.nonmedia / (1 - refresh_steps / rollingWindowLength)
          }

          if (is.nan(decomp.rssd)) {
            # message("all media in this iteration have 0 coefficients")
            decomp.rssd <- Inf
            dt_decompSpendDist[, effect_share := 0]
          }

          ## adstock objective: sum of squared infinite sum of decay to be minimised - deprecated
          # dt_decaySum <- dt_mediaVecCum[,  .(rn = all_media, decaySum = sapply(.SD, sum)), .SDcols = all_media]
          # adstock.ssisd <- dt_decaySum[, sum(decaySum^2)]

          ## calibration objective: not calibration: mse, decomp.rssd, if calibration: mse, decom.rssd, mape_lift

          #####################################
          #### Collect output

          resultHypParam <- data.table()[, (hypParamSamName) := lapply(hypParamSam[1:length(hypParamSamName)], function(x) x)]

          resultCollect <- list(
            resultHypParam = resultHypParam[, ":="(
              mape = mape,
              nrmse = nrmse,
              decomp.rssd = decomp.rssd
              # ,adstock.ssisd = adstock.ssisd
              , rsq_train = mod_out$rsq_train
              # ,rsq_test = mod_out$rsq_test
              , pos = prod(decompCollect$xDecompAgg$pos),
              lambda = lambda
              # ,Score = -mape
              , Elapsed = as.numeric(difftime(Sys.time(), t1, units = "secs")),
              ElapsedAccum = as.numeric(difftime(Sys.time(), t0, units = "secs")),
              iterPar = i,
              iterNG = lng,
              df.int = df.int)],
            xDecompVec = if (hyper_fixed == TRUE) {
              decompCollect$xDecompVec[, ":="(
                intercept = decompCollect$xDecompAgg[rn == "(Intercept)", xDecompAgg],
                mape = mape,
                nrmse = nrmse,
                decomp.rssd = decomp.rssd
                # ,adstock.ssisd = adstock.ssisd
                , rsq_train = mod_out$rsq_train
                # ,rsq_test = mod_out$rsq_test
                , lambda = lambda,
                iterPar = i,
                iterNG = lng,
                df.int = df.int)]
            } else {
              NULL
            },
            xDecompAgg = decompCollect$xDecompAgg[, ":="(
              mape = mape,
              nrmse = nrmse,
              decomp.rssd = decomp.rssd
              # ,adstock.ssisd = adstock.ssisd
              , rsq_train = mod_out$rsq_train
              # ,rsq_test = mod_out$rsq_test
              , lambda = lambda,
              iterPar = i,
              iterNG = lng,
              df.int = df.int)],
            liftCalibration = if (!is.null(calibration_input)) {
              liftCollect[, ":="(
                mape = mape,
                nrmse = nrmse,
                decomp.rssd = decomp.rssd
                # ,adstock.ssisd = adstock.ssisd
                , rsq_train = mod_out$rsq_train
                # ,rsq_test = mod_out$rsq_test
                , lambda = lambda,
                iterPar = i,
                iterNG = lng)]
            } else {
              NULL
            },
            decompSpendDist = dt_decompSpendDist[, ":="(
              mape = mape,
              nrmse = nrmse,
              decomp.rssd = decomp.rssd
              # ,adstock.ssisd = adstock.ssisd
              , rsq_train = mod_out$rsq_train
              # ,rsq_test = mod_out$rsq_test
              , lambda = lambda,
              iterPar = i,
              iterNG = lng,
              df.int = df.int)],
            mape.lift = mape,
            nrmse = nrmse,
            decomp.rssd = decomp.rssd,
            iterPar = i,
            iterNG = lng,
            df.int = df.int
            # ,cvmod = cvmod
          )

          best_mape <- min(best_mape, mape)
          if (cnt == iterTotal) {
            print(" === ")
            print(paste0("Optimizer_name: ", optimizer_name, ";  Total_iterations: ", cnt, ";   best_mape: ", best_mape))
          }
          return(resultCollect)
        }
      ) # end foreach parallel

      nrmse.collect <- sapply(doparCollect, function(x) x$nrmse)
      decomp.rssd.collect <- sapply(doparCollect, function(x) x$decomp.rssd)
      mape.lift.collect <- sapply(doparCollect, function(x) x$mape.lift)

      #####################################
      #### Nevergrad tells objectives

      if (hyper_fixed == FALSE) {
        if (is.null(calibration_input)) {
          for (co in 1:iterPar) {
            optimizer$tell(nevergrad_hp[[co]], tuple(nrmse.collect[co], decomp.rssd.collect[co]))
          }
        } else {
          for (co in 1:iterPar) {
            optimizer$tell(nevergrad_hp[[co]], tuple(nrmse.collect[co], decomp.rssd.collect[co], mape.lift.collect[co]))
          }
        }
      }

      resultCollectNG[[lng]] <- doparCollect
      cnt <- cnt + iterPar
      if (hyper_fixed == FALSE) setTxtProgressBar(pb, cnt)
    } ## end NG loop
  }) # end system.time

  cat("\r", paste("\n  Finished in", round(sysTimeDopar[3] / 60, 2), "mins"))
  flush.console()

  # stop cluster to avoid memory leaks
  stopImplicitCluster()

  if (hyper_fixed == FALSE) close(pb)
  registerDoSEQ()
  getDoParWorkers()

  #####################################
  #### Final result collect

  resultCollect <- list(
    resultHypParam = rbindlist(lapply(resultCollectNG, function(x) {
      rbindlist(lapply(x, function(y) y$resultHypParam))
    }))[order(nrmse)],
    xDecompVec = if (hyper_fixed == TRUE) {
      rbindlist(lapply(resultCollectNG, function(x) {
        rbindlist(lapply(x, function(y) y$xDecompVec))
      }))[order(nrmse, ds)]
    } else {
      NULL
    },
    xDecompAgg = rbindlist(lapply(resultCollectNG, function(x) {
      rbindlist(lapply(x, function(y) y$xDecompAgg))
    }))[order(nrmse)],
    liftCalibration = if (!is.null(calibration_input)) {
      rbindlist(lapply(resultCollectNG, function(x) {
        rbindlist(lapply(x, function(y) y$liftCalibration))
      }))[order(mape, liftMedia, liftStart)]
    } else {
      NULL
    },
    decompSpendDist = rbindlist(lapply(resultCollectNG, function(x) {
      rbindlist(lapply(x, function(y) y$decompSpendDist))
    }))[order(nrmse)]
  )
  resultCollect$iter <- length(resultCollect$mape)
  resultCollect$elapsed.min <- sysTimeDopar[3] / 60
  # Adjust accumulated time
  resultCollect$resultHypParam[, ElapsedAccum := ElapsedAccum - min(ElapsedAccum) + resultCollect$resultHypParam[which.min(ElapsedAccum), Elapsed]]

  return(list(
    resultCollect = resultCollect,
    hyperBoundNG = hyper_bound_list_updated,
    hyperBoundFixed = hyper_bound_list_fixed
  ))
}

####################################################################
#' The response function
#'
#' The \code{robyn_response()} function returns the response for a given
#' spend level of a given \code{paid_media_vars} from a selected model
#' result from a selected model build (initial model, refresh model etc.).
#'
#' @inheritParams robyn_allocator
#' @param paid_media_var A character. Selected paid media variable for the response.
#' Must be within \code{InputCollect$paid_media_vars}
#' @param spend Numeric. The desired spend level to return a response for.
#' @param dt_hyppar A data.table. When \code{robyn_object} is not provided, use
#' \code{dt_hyppar = OutputCollect$resultHypParam}. It must be provided along
#' \code{select_model}, \code{dt_coef} and \code{InputCollect}.
#' @param dt_coef A data.table. When \code{robyn_object} is not provided, use
#' \code{dt_coef = OutputCollect$xDecompAgg}. It must be provided along
#' \code{select_model}, \code{dt_hyppar} and \code{InputCollect}.
#' @examples
#' \dontrun{
#' ## Get marginal response (mResponse) and marginal ROI (mROI) for
#' ## the next 1k on 80k for search_clicks_P, when provided the saved
#' ## robyn_object by the robyn_save() function.
#'
#' # Get response for 80k
#' spend1 <- 80000
#' Response1 <- robyn_response(
#'   robyn_object = robyn_object,
#'   paid_media_var = "search_clicks_P",
#'   spend = spend1
#' )
#'
#' # Get ROI for 80k
#' Response1 / spend1 # ROI for search 80k
#'
#' # Get response for 81k
#' spend2 <- spend1 + 1000
#' Response2 <- robyn_response(
#'   robyn_object = robyn_object,
#'   paid_media_var = "search_clicks_P",
#'   spend = spend2
#' )
#'
#' # Get ROI for 81k
#' Response2 / spend2 # ROI for search 81k
#'
#' # Get marginal response (mResponse) for the next 1k on 80k
#' Response2 - Response1
#'
#' # Get marginal ROI (mROI) for the next 1k on 80k
#' (Response2 - Response1) / (spend2 - spend1)
#'
#'
#' ## Get response for 80k for search_clicks_P from the third model refresh
#'
#' robyn_response(
#'   robyn_object = robyn_object,
#'   select_build = 3,
#'   paid_media_var = "search_clicks_P",
#'   spend = 80000
#' )
#'
#' ## Get response for 80k for search_clicks_P from the a certain model SolID
#' ## in the current model output in the global environment
#'
#' robyn_response(,
#'   paid_media_var = "search_clicks_P",
#'   select_model = "3_10_3",
#'   spend = 80000,
#'   dt_hyppar = OutputCollect$resultHypParam,
#'   dt_coef = OutputCollect$xDecompAgg,
#'   InputCollect = InputCollect
#' )
#' }
#' @export
robyn_response <- function(robyn_object = NULL,
                           select_build = NULL,
                           paid_media_var = NULL,
                           select_model = NULL,
                           spend = NULL,
                           dt_hyppar = NULL,
                           dt_coef = NULL,
                           InputCollect = NULL) {

  ## get input
  if (!is.null(robyn_object)) {

    if (!file.exists(robyn_object)) {
      stop("File does not exist or is somewhere else. Check: ", robyn_object)
    } else {
      Robyn <- readRDS(robyn_object)
      objectPath <- dirname(robyn_object)
      objectName <- sub("'\\..*$", "", basename(robyn_object))
    }

    select_build_all <- 0:(length(Robyn) - 1)
    if (is.null(select_build)) {
      select_build <- max(select_build_all)
      message(
        "Using latest model: ", ifelse(select_build == 0, "initial model", paste0("refresh model nr.", select_build)),
        " for the response function. Use parameter 'select_build' to specify which run to use"
      )
    }

    if (!(select_build %in% select_build_all) | length(select_build) != 1) {
      stop("select_build must be one value of ", paste(select_build_all, collapse = ", "))
    }

    listName <- ifelse(select_build == 0, "listInit", paste0("listRefresh", select_build))
    InputCollect <- Robyn[[listName]][["InputCollect"]]
    OutputCollect <- Robyn[[listName]][["OutputCollect"]]
    dt_hyppar <- OutputCollect$resultHypParam
    dt_coef <- OutputCollect$xDecompAgg
    select_model <- OutputCollect$selectID
  } else if (any(is.null(dt_hyppar), is.null(dt_coef), is.null(InputCollect))) {
    stop(paste(
      "When 'robyn_object' is not provided, then 'dt_hyppar = OutputCollect$resultHypParam',",
      "'dt_coef = OutputCollect$xDecompAgg' and 'InputCollect' must be provided"
    ))
  }

  dt_input <- InputCollect$dt_input
  paid_media_vars <- InputCollect$paid_media_vars
  paid_media_spends <- InputCollect$paid_media_spends
  startRW <- InputCollect$rollingWindowStartWhich
  endRW <- InputCollect$rollingWindowEndWhich
  adstock <- InputCollect$adstock
  allSolutions <- dt_hyppar[, unique(solID)]
  spendExpoMod <- InputCollect$modNLSCollect

  ## check inputs
  if (is.null(paid_media_var)) {
    stop(paste0("paid_media_var must be one of these values: ", paste(paid_media_vars, collapse = ", ")))
  } else if (!(paid_media_var %in% paid_media_vars) | length(paid_media_var) != 1) {
    stop(paste0("paid_media_var must be one of these values: ", paste(paid_media_vars, collapse = ", ")))
  }

  if (!(select_model %in% allSolutions)) {
    stop(paste0("select_model must be one of these values: ", paste(allSolutions, collapse = ", ")))
  }

  mediaVar <- dt_input[, get(paid_media_var)]

  if (!is.null(spend)) {
    if (length(spend) != 1 | spend <= 0 | !is.numeric(spend)) {
      stop("'spend' must be a positive number")
    }
  }

  ## transform spend to exposure if necessary
  if (paid_media_var %in% InputCollect$exposureVarName) {

    # use non-0 mean spend as marginal level if spend not provided
    if (is.null(spend)) {
      mediaspend <- dt_input[startRW:endRW, get(paid_media_spends[which(paid_media_vars == paid_media_var)])]
      spend <- mean(mediaspend[mediaspend > 0])
      message("'spend' not provided. Using mean of ", paid_media_var, " as marginal level instead")
    }

    # fit spend to exposure
    nls_select <- spendExpoMod[channel == paid_media_var, rsq_nls > rsq_lm]
    if (nls_select) {
      Vmax <- spendExpoMod[channel == paid_media_var, Vmax]
      Km <- spendExpoMod[channel == paid_media_var, Km]
      spend <- mic_men(x = spend, Vmax = Vmax, Km = Km, reverse = FALSE)
    } else {
      coef_lm <- spendExpoMod[channel == paid_media_var, coef_lm]
      spend <- spend * coef_lm
    }
  } else {

    # use non-0 mean spend as marginal level if spend not provided
    if (is.null(spend)) {
      mediaspend <- dt_input[startRW:endRW, get(paid_media_var)]
      spend <- mean(mediaspend[mediaspend > 0])
      message("spend not provided. using mean of ", paid_media_var, " as marginal levl instead")
    }
  }


  ## adstocking
  if (adstock == "geometric") {
    theta <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_thetas"))]
    x_list <- adstock_geometric(x = mediaVar, theta = theta)
  } else if (adstock == "weibull_cdf") {
    shape <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_shapes"))]
    scale <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_scales"))]
    x_list <- adstock_weibull(x = mediaVar, shape = shape, scale = scale, windlen = InputCollect$rollingWindowLength, type = "cdf")
  } else if (adstock == "weibull_pdf") {
    shape <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_shapes"))]
    scale <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_scales"))]
    x_list <- adstock_weibull(x = mediaVar, shape = shape, scale = scale, windlen = InputCollect$rollingWindowLength, type = "pdf")
  }
  m_adstocked <- x_list$x_decayed

  ## saturation
  m_adstockedRW <- m_adstocked[startRW:endRW]
  alpha <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_alphas"))]
  gamma <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_gammas"))]
  Saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma, x_marginal = spend)

  ## decomp
  coeff <- dt_coef[solID == select_model & rn == paid_media_var, coef]
  Response <- Saturated * coeff

  return(as.numeric(Response))
}


model_decomp <- function(coefs, dt_modSaturated, x, y_pred, i, dt_modRollWind, refreshAddedStart) {

  ## input for decomp
  y <- dt_modSaturated$dep_var
  indepVar <- dt_modSaturated[, (setdiff(names(dt_modSaturated), "dep_var")), with = FALSE]
  x <- as.data.table(x)
  intercept <- coefs[1]
  indepVarName <- names(indepVar)
  indepVarCat <- indepVarName[sapply(indepVar, is.factor)]

  ## decomp x
  xDecomp <- data.table(mapply(function(regressor, coeff) {
    regressor * coeff
  }, regressor = x, coeff = coefs[-1]))
  xDecomp <- cbind(data.table(intercept = rep(intercept, nrow(xDecomp))), xDecomp)
  # xDecompOut <- data.table(sapply(indepVarName, function(x) xDecomp[, rowSums(.SD,), .SDcols = str_which(names(xDecomp), x)]))
  xDecompOut <- cbind(data.table(ds = dt_modRollWind$ds, y = y, y_pred = y_pred), xDecomp)

  ## QA decomp
  y_hat <- rowSums(xDecomp)
  errorTerm <- y_hat - y_pred
  if (prod(round(y_pred) == round(y_hat)) == 0) {
    message("\n### attention for loop ", i, " : manual decomp is not matching linear model prediction. Deviation is ", mean(errorTerm / y) * 100, " % ### \n")
  }

  ## output decomp
  y_hat.scaled <- rowSums(abs(xDecomp))
  xDecompOutPerc.scaled <- abs(xDecomp) / y_hat.scaled
  xDecompOut.scaled <- y_hat * xDecompOutPerc.scaled

  xDecompOutAgg <- sapply(xDecompOut[, c("intercept", indepVarName), with = FALSE], function(x) sum(x))
  xDecompOutAggPerc <- xDecompOutAgg / sum(y_hat)
  xDecompOutAggMeanNon0 <- sapply(xDecompOut[, c("intercept", indepVarName), with = FALSE], function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x != 0])))
  xDecompOutAggMeanNon0[is.nan(xDecompOutAggMeanNon0)] <- 0
  xDecompOutAggMeanNon0Perc <- xDecompOutAggMeanNon0 / sum(xDecompOutAggMeanNon0)
  # xDecompOutAggPerc.scaled <- abs(xDecompOutAggPerc)/sum(abs(xDecompOutAggPerc))
  # xDecompOutAgg.scaled <- sum(xDecompOutAgg)*xDecompOutAggPerc.scaled

  refreshAddedStartWhich <- which(xDecompOut$ds == refreshAddedStart)
  refreshAddedEnd <- max(xDecompOut$ds)
  refreshAddedEndWhich <- which(xDecompOut$ds == refreshAddedEnd)
  xDecompOutAggRF <- sapply(xDecompOut[refreshAddedStartWhich:refreshAddedEndWhich, c("intercept", indepVarName), with = FALSE], function(x) sum(x))
  y_hatRF <- y_hat[refreshAddedStartWhich:refreshAddedEndWhich]
  xDecompOutAggPercRF <- xDecompOutAggRF / sum(y_hatRF)
  xDecompOutAggMeanNon0RF <- sapply(xDecompOut[refreshAddedStartWhich:refreshAddedEndWhich, c("intercept", indepVarName), with = FALSE], function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x != 0])))
  xDecompOutAggMeanNon0RF[is.nan(xDecompOutAggMeanNon0RF)] <- 0
  xDecompOutAggMeanNon0PercRF <- xDecompOutAggMeanNon0RF / sum(xDecompOutAggMeanNon0RF)

  coefsOut <- data.table(coefs, keep.rownames = TRUE)
  coefsOutCat <- copy(coefsOut)
  coefsOut[, rn := if (length(indepVarCat) == 0) {
    rn
  } else {
    sapply(indepVarCat, function(x) str_replace(coefsOut$rn, paste0(x, ".*"), x))
  }]
  coefsOut <- coefsOut[, .(coef = mean(s0)), by = rn]

  decompOutAgg <- cbind(coefsOut, data.table(
    xDecompAgg = xDecompOutAgg,
    xDecompPerc = xDecompOutAggPerc,
    xDecompMeanNon0 = xDecompOutAggMeanNon0,
    xDecompMeanNon0Perc = xDecompOutAggMeanNon0Perc,
    xDecompAggRF = xDecompOutAggRF,
    xDecompPercRF = xDecompOutAggPercRF,
    xDecompMeanNon0RF = xDecompOutAggMeanNon0RF,
    xDecompMeanNon0PercRF = xDecompOutAggMeanNon0PercRF
    # ,xDecompAgg.scaled = xDecompOutAgg.scaled
    # ,xDecompPerc.scaled = xDecompOutAggPerc.scaled
  ))
  decompOutAgg[, pos := xDecompAgg >= 0]

  decompCollect <- list(xDecompVec = xDecompOut, xDecompVec.scaled = xDecompOut.scaled, xDecompAgg = decompOutAgg, coefsOutCat = coefsOutCat)

  return(decompCollect)
} ## decomp end


calibrate_mmm <- function(decompCollect, calibration_input, paid_media_vars, dayInterval) {

  # check if any lift channel doesn't have media var
  check_set_lift <- any(sapply(calibration_input$channel, function(x) {
    any(str_detect(x, paid_media_vars))
  }) == FALSE)
  if (check_set_lift) {
    stop("calibration_input channels must have media variable")
  }

  ## prep lift input
  getLiftMedia <- unique(calibration_input$channel)
  getDecompVec <- decompCollect$xDecompVec

  ## loop all lift input
  liftCollect <- list()
  for (m in 1:length(getLiftMedia)) { # loop per lift channel

    liftWhich <- str_which(calibration_input$channel, getLiftMedia[m])

    liftCollect2 <- list()
    for (lw in 1:length(liftWhich)) { # loop per lift test per channel

      ## get lift period subset
      liftStart <- calibration_input[liftWhich[lw], liftStartDate]
      liftEnd <- calibration_input[liftWhich[lw], liftEndDate]
      liftPeriodVec <- getDecompVec[ds >= liftStart & ds <= liftEnd, c("ds", getLiftMedia[m]), with = FALSE]
      liftPeriodVecDependent <- getDecompVec[ds >= liftStart & ds <= liftEnd, c("ds", "y"), with = FALSE]

      ## scale decomp
      mmmDays <- nrow(liftPeriodVec) * dayInterval
      liftDays <- as.integer(liftEnd - liftStart + 1)
      y_hatLift <- sum(unlist(getDecompVec[, -1])) # total pred sales
      x_decompLift <- sum(liftPeriodVec[, 2])
      x_decompLiftScaled <- x_decompLift / mmmDays * liftDays
      y_scaledLift <- liftPeriodVecDependent[, sum(y)] / mmmDays * liftDays

      ## output
      liftCollect2[[lw]] <- data.table(
        liftMedia = getLiftMedia[m],
        liftStart = liftStart,
        liftEnd = liftEnd,
        liftAbs = calibration_input[liftWhich[lw], liftAbs],
        decompAbsScaled = x_decompLiftScaled,
        dependent = y_scaledLift
      )
    }
    liftCollect[[m]] <- rbindlist(liftCollect2)
  }

  ## get mape_lift
  liftCollect <- rbindlist(liftCollect)[, mape_lift := abs((decompAbsScaled - liftAbs) / liftAbs)]
  return(liftCollect)
}


model_refit <- function(x_train, y_train, lambda, lower.limits, upper.limits) {
  mod <- glmnet(
    x_train,
    y_train,
    family = "gaussian",
    alpha = 0, # 0 for ridge regression
    # https://stats.stackexchange.com/questions/138569/why-is-lambda-within-one-standard-error-from-the-minimum-is-a-recommended-valu
    lambda = lambda,
    lower.limits = lower.limits,
    upper.limits = upper.limits
  ) # coef(mod)

  ## drop intercept if negative
  if (coef(mod)[1] < 0) {
    mod <- glmnet(
      x_train,
      y_train,
      family = "gaussian",
      alpha = 0 # 0 for ridge regression
      , lambda = lambda,
      lower.limits = lower.limits,
      upper.limits = upper.limits,
      intercept = FALSE
    ) # coef(mod)
  } # ; plot(mod); print(mod)

  df.int <- ifelse(coef(mod)[1] < 0, 0, 1)

  y_trainPred <- predict(mod, s = lambda, newx = x_train)
  rsq_train <- get_rsq(true = y_train, predicted = y_trainPred, p = ncol(x_train), df.int = df.int)
  rsq_train

  # y_testPred <- predict(mod, s = lambda, newx = x_test)
  # rsq_test <- get_rsq(true = y_test, predicted = y_testPred); rsq_test

  # mape_mod<- mean(abs((y_test - y_testPred)/y_test)* 100); mape_mod
  coefs <- as.matrix(coef(mod))
  # y_pred <- c(y_trainPred, y_testPred)

  # mean(y_train) sd(y_train)
  nrmse_train <- sqrt(mean((y_train - y_trainPred)^2)) / (max(y_train) - min(y_train))
  # nrmse_test <- sqrt(mean(sum((y_test - y_testPred)^2))) /
  # (max(y_test) - min(y_test)) # mean(y_test) sd(y_test)

  mod_out <- list(
    rsq_train = rsq_train
    # ,rsq_test = rsq_test
    , nrmse_train = nrmse_train
    # ,nrmse_test = nrmse_test
    # ,mape_mod = mape_mod
    , coefs = coefs,
    y_pred = y_trainPred,
    mod = mod,
    df.int = df.int
  )

  return(mod_out)
}

ridge_lambda <- function(x, y, seq_len = 100, lambda_min_ratio = 0.0001) {
  mysd <- function(y) sqrt(sum((y - mean(y))^2) / length(y))
  sx <- scale(x, scale = apply(x, 2, mysd))
  sx <- as.matrix(sx, ncol = ncol(x), nrow = nrow(x))
  # sy <- as.vector(scale(y, scale=mysd(y)))
  sy <- y
  # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)
  lambda_max <- max(abs(colSums(sx * sy))) / (0.001 * nrow(x))
  lambda_max_log <- log(lambda_max)
  log_step <- (log(lambda_max) - log(lambda_max * lambda_min_ratio)) / (seq_len - 1)
  log_seq <- seq(log(lambda_max), log(lambda_max * lambda_min_ratio), length.out = seq_len)
  lambda_seq <- exp(log_seq)
  return(lambda_seq)
}
