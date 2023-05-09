# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn Modelling Function
#'
#' \code{robyn_run()} consumes \code{robyn_input()} outputs,
#' runs \code{robyn_mmm()}, and collects all modeling results.
#'
#' @inheritParams robyn_allocator
#' @inheritParams robyn_outputs
#' @inheritParams robyn_inputs
#' @param dt_hyper_fixed data.frame or named list. Only provide when loading
#' old model results. It consumes hyperparameters from saved csv
#' \code{pareto_hyperparameters.csv} or JSON file to replicate a model.
#' @param ts_validation Boolean. When set to \code{TRUE}, Robyn will split data
#' by test, train, and validation partitions to validate the time series. By
#' default the "train_size" range is set to \code{c(0.5, 0.8)}, but it can be
#' customized or set to a fixed value using the hyperparameters input. For example,
#' if \code{train_size = 0.7}, validation size and test size will both be 0.15
#' and 0.15. When \code{ts_validation = FALSE}, nrmse_train is the
#' objective function; when \code{ts_validation = TRUE}, nrmse_val is the objective
#' function.
#' @param add_penalty_factor Boolean. Add penalty factor hyperparameters to
#' glmnet's penalty.factor to be optimized by nevergrad. Use with caution, because
#' this feature might add too much hyperparameter space and probably requires
#' more iterations to converge.
#' @param refresh Boolean. Set to \code{TRUE} when used in \code{robyn_refresh()}.
#' @param cores Integer. Default to \code{parallel::detectCores() - 1} (all cores
#' except one). Set to 1 if you want to turn parallel computing off.
#' @param iterations Integer. Recommended 2000 for default when using
#' \code{nevergrad_algo = "TwoPointsDE"}.
#' @param trials Integer. Recommended 5 for default
#' \code{nevergrad_algo = "TwoPointsDE"}.
#' @param nevergrad_algo Character. Default to "TwoPointsDE". Options are
#' \code{c("DE","TwoPointsDE", "OnePlusOne", "DoubleFastGADiscreteOnePlusOne",
#' "DiscreteOnePlusOne", "PortfolioDiscreteOnePlusOne", "NaiveTBPSA",
#' "cGA", "RandomSearch")}.
#' @param intercept Boolean. Should intercept(s) be fitted (default=TRUE) or
#' set to zero (FALSE).
#' @param intercept_sign Character. Choose one of "non_negative" (default) or
#' "unconstrained". By default, if intercept is negative, Robyn will drop intercept
#' and refit the model. Consider changing intercept_sign to "unconstrained" when
#' there are \code{context_vars} with large positive values.
#' @param rssd_zero_penalty Boolean. When TRUE, the objective function
#' DECOMP.RSSD will penalize models with more 0 media effects additionally.
#' In other words, given the same DECOMP.RSSD score, a model with 50\% 0-coef
#' variables will get penalized by DECOMP.RSSD * 1.5 (larger error), while
#' another model with no 0-coef variables gets un-penalized with DECOMP.RSSD * 1.
#' @param seed Integer. For reproducible results when running nevergrad.
#' @param outputs Boolean. Process results with \code{robyn_outputs()}?
#' @param lambda_control Deprecated in v3.6.0.
#' @param ... Additional parameters passed to \code{robyn_outputs()}.
#' @return List. Class: \code{robyn_models}. Contains the results of all trials
#' and iterations modeled.
#' @examples
#' \dontrun{
#' # Having InputCollect results
#' OutputCollect <- robyn_run(
#'   InputCollect = InputCollect,
#'   cores = 2,
#'   iterations = 200,
#'   trials = 1,
#'   outputs = FALSE
#' )
#' }
#' @return List. Contains all trained models. Class: \code{robyn_models}.
#' @export
robyn_run <- function(InputCollect = NULL,
                      dt_hyper_fixed = NULL,
                      json_file = NULL,
                      ts_validation = FALSE,
                      add_penalty_factor = FALSE,
                      refresh = FALSE,
                      seed = 123L,
                      outputs = FALSE,
                      quiet = FALSE,
                      cores = NULL,
                      trials = 5,
                      iterations = 2000,
                      rssd_zero_penalty = TRUE,
                      nevergrad_algo = "TwoPointsDE",
                      intercept = TRUE,
                      intercept_sign = "non_negative",
                      lambda_control = NULL,
                      ...) {
  t0 <- Sys.time()

  ### Use previously exported model using json_file
  if (!is.null(json_file)) {
    # InputCollect <- robyn_inputs(json_file = json_file, dt_input = dt_input, dt_holidays = dt_holidays)
    if (is.null(InputCollect)) InputCollect <- robyn_inputs(json_file = json_file, ...)
    json <- robyn_read(json_file, step = 2, quiet = TRUE)
    dt_hyper_fixed <- json$ExportedModel$hyper_values
    for (i in seq_along(json$ExportedModel)) {
      assign(names(json$ExportedModel)[i], json$ExportedModel[[i]])
    }
    bootstrap <- select(json$ExportedModel$summary, any_of(c("variable", "boot_mean", "ci_low", "ci_up")))
    if (is.null(seed) | length(seed) == 0) seed <- 123L
    dt_hyper_fixed$solID <- json$ExportedModel$select_model
  } else {
    bootstrap <- NULL
  }

  #####################################
  #### Set local environment

  if (!"hyperparameters" %in% names(InputCollect) || is.null(InputCollect$hyperparameters)) {
    stop("Must provide 'hyperparameters' in robyn_inputs()'s output first")
  }

  # Check and warn on legacy inputs (using InputCollect params as robyn_run() inputs)
  InputCollect <- check_legacy_input(InputCollect, cores, iterations, trials, intercept_sign, nevergrad_algo)
  # Overwrite values imported from InputCollect
  legacyValues <- InputCollect[LEGACY_PARAMS]
  legacyValues <- legacyValues[!unlist(lapply(legacyValues, is.null))]
  if (length(legacyValues) > 0) {
    for (i in seq_along(InputCollect)) assign(names(InputCollect)[i], InputCollect[[i]])
  }

  # Keep in mind: https://www.jottr.org/2022/12/05/avoid-detectcores/
  max_cores <- max(1L, parallel::detectCores(), na.rm = TRUE)
  if (is.null(cores)) {
    cores <- max_cores - 1 # It's recommended to always leave at least one core free
  } else if (cores > max_cores) {
    warning(sprintf("Max possible cores in your machine is %s (your input was %s)", max_cores, cores))
    cores <- max_cores
  }
  if (cores == 0) cores <- 1

  hyps_fixed <- !is.null(dt_hyper_fixed)
  if (hyps_fixed) trials <- iterations <- 1
  check_run_inputs(cores, iterations, trials, intercept_sign, nevergrad_algo)
  check_iteration(InputCollect$calibration_input, iterations, trials, hyps_fixed, refresh)
  init_msgs_run(InputCollect, refresh, lambda_control = NULL, quiet)

  #####################################
  #### Prepare hyper-parameters
  hyper_collect <- hyper_collector(
    InputCollect,
    hyper_in = InputCollect$hyperparameters,
    ts_validation = ts_validation,
    add_penalty_factor = add_penalty_factor,
    dt_hyper_fixed = dt_hyper_fixed,
    cores = cores
  )
  InputCollect$hyper_updated <- hyper_collect$hyper_list_all

  #####################################
  #### Run robyn_mmm() for each trial

  OutputModels <- robyn_train(
    InputCollect, hyper_collect,
    cores = cores, iterations = iterations, trials = trials,
    intercept_sign = intercept_sign, intercept = intercept,
    nevergrad_algo = nevergrad_algo,
    dt_hyper_fixed = dt_hyper_fixed,
    ts_validation = ts_validation,
    add_penalty_factor = add_penalty_factor,
    rssd_zero_penalty = rssd_zero_penalty,
    refresh, seed, quiet
  )

  attr(OutputModels, "hyper_fixed") <- hyper_collect$all_fixed
  attr(OutputModels, "bootstrap") <- bootstrap
  attr(OutputModels, "refresh") <- refresh

  if (TRUE) {
    OutputModels$cores <- cores
    OutputModels$iterations <- iterations
    OutputModels$trials <- trials
    OutputModels$intercept <- intercept
    OutputModels$intercept_sign <- intercept_sign
    OutputModels$nevergrad_algo <- nevergrad_algo
    OutputModels$ts_validation <- ts_validation
    OutputModels$add_penalty_factor <- add_penalty_factor
    OutputModels$hyper_updated <- hyper_collect$hyper_list_all
  }

  # Not direct output & not all fixed hyperparameters
  if (!outputs & is.null(dt_hyper_fixed)) {
    output <- OutputModels
  } else if (!hyper_collect$all_fixed) {
    # Direct output & not all fixed hyperparameters, including refresh mode
    output <- robyn_outputs(InputCollect, OutputModels, refresh = refresh, ...)
  } else {
    # Direct output & all fixed hyperparameters, thus no cluster
    output <- robyn_outputs(InputCollect, OutputModels, clusters = FALSE, ...)
  }

  # Check convergence when more than 1 iteration
  if (!hyper_collect$all_fixed) {
    output[["convergence"]] <- robyn_converge(OutputModels, ...)
    output[["ts_validation_plot"]] <- ts_validation(OutputModels, ...)
  } else {
    if ("solID" %in% names(dt_hyper_fixed)) {
      output[["selectID"]] <- dt_hyper_fixed$solID
    } else {
      output[["selectID"]] <- OutputModels$trial1$resultCollect$resultHypParam$solID
    }
    if (!quiet) message("Successfully recreated model ID: ", output$selectID)
  }

  # Save hyper-parameters list
  output[["hyper_updated"]] <- hyper_collect$hyper_list_all
  output[["seed"]] <- seed

  # Report total timing
  attr(output, "runTime") <- round(difftime(Sys.time(), t0, units = "mins"), 2)
  if (!quiet && iterations > 1) message(paste("Total run time:", attr(output, "runTime"), "mins"))

  class(output) <- unique(c("robyn_models", class(output)))
  return(output)
}

#' @rdname robyn_run
#' @aliases robyn_run
#' @param x \code{robyn_models()} output.
#' @export
print.robyn_models <- function(x, ...) {
  is_fixed <- all(lapply(x$hyper_updated, length) == 1)
  print(glued(
    "
  Total trials: {x$trials}
  Iterations per trial: {x$iterations} {total_iters}
  Runtime (minutes): {attr(x, 'runTime')}
  Cores: {x$cores}

  Updated Hyper-parameters{fixed}:
  {hypers}

  Nevergrad Algo: {x$nevergrad_algo}
  Intercept: {x$intercept}
  Intercept sign: {x$intercept_sign}
  Time-series validation: {x$ts_validation}
  Penalty factor: {x$add_penalty_factor}
  Refresh: {isTRUE(attr(x, 'refresh'))}

  Convergence on last quantile (iters {iters}):
    {convergence}

  ",
    total_iters = sprintf("(%s real)", ifelse(
      "trial1" %in% names(x), nrow(x$trial1$resultCollect$resultHypParam), 1
    )),
    iters = ifelse(is.null(x$convergence), 1, paste(tail(x$convergence$errors$cuts, 2), collapse = ":")),
    fixed = ifelse(is_fixed, " (fixed)", ""),
    convergence = if (!is_fixed) paste(x$convergence$conv_msg, collapse = "\n  ") else "Fixed hyper-parameters",
    hypers = flatten_hyps(x$hyper_updated)
  ))

  if ("robyn_outputs" %in% class(x)) {
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
}

####################################################################
#' Train Robyn Models
#'
#' \code{robyn_train()} consumes output from \code{robyn_input()}
#' and runs the \code{robyn_mmm()} on each trial.
#'
#' @inheritParams robyn_run
#' @param hyper_collect List. Containing hyperparameter bounds. Defaults to
#' \code{InputCollect$hyperparameters}.
#' @return List. Iteration results to include in \code{robyn_run()} results.
#' @export
robyn_train <- function(InputCollect, hyper_collect,
                        cores, iterations, trials,
                        intercept_sign, intercept,
                        nevergrad_algo,
                        dt_hyper_fixed = NULL,
                        ts_validation = TRUE,
                        add_penalty_factor = FALSE,
                        rssd_zero_penalty = TRUE,
                        refresh = FALSE, seed = 123,
                        quiet = FALSE) {
  hyper_fixed <- hyper_collect$all_fixed

  if (hyper_fixed) {
    OutputModels <- list()
    OutputModels[[1]] <- robyn_mmm(
      InputCollect = InputCollect,
      hyper_collect = hyper_collect,
      iterations = iterations,
      cores = cores,
      nevergrad_algo = nevergrad_algo,
      intercept = intercept,
      intercept_sign = intercept_sign,
      dt_hyper_fixed = dt_hyper_fixed,
      ts_validation = ts_validation,
      add_penalty_factor = add_penalty_factor,
      rssd_zero_penalty = rssd_zero_penalty,
      seed = seed,
      quiet = quiet
    )
    OutputModels[[1]]$trial <- 1
    # Set original solID (to overwrite default 1_1_1)
    if ("solID" %in% names(dt_hyper_fixed)) {
      these <- c("resultHypParam", "xDecompVec", "xDecompAgg", "decompSpendDist")
      for (tab in these) OutputModels[[1]]$resultCollect[[tab]]$solID <- dt_hyper_fixed$solID
    }
  } else {
    ## Run robyn_mmm() for each trial if hyperparameters are not all fixed
    check_init_msg(InputCollect, cores)
    if (!quiet) {
      message(paste(
        ">>> Starting", trials, "trials with",
        iterations, "iterations each",
        ifelse(is.null(InputCollect$calibration_input), "using", "with calibration using"),
        nevergrad_algo, "nevergrad algorithm..."
      ))
    }

    OutputModels <- list()

    for (ngt in 1:trials) { # ngt = 1
      if (!quiet) message(paste("  Running trial", ngt, "of", trials))
      model_output <- robyn_mmm(
        InputCollect = InputCollect,
        hyper_collect = hyper_collect,
        iterations = iterations,
        cores = cores,
        nevergrad_algo = nevergrad_algo,
        intercept = intercept,
        intercept_sign = intercept_sign,
        ts_validation = ts_validation,
        add_penalty_factor = add_penalty_factor,
        rssd_zero_penalty = rssd_zero_penalty,
        refresh = refresh,
        trial = ngt,
        seed = seed + ngt,
        quiet = quiet
      )
      check_coef0 <- any(model_output$resultCollect$decompSpendDist$decomp.rssd == Inf)
      if (check_coef0) {
        num_coef0_mod <- filter(model_output$resultCollect$decompSpendDist, is.infinite(.data$decomp.rssd)) %>%
          distinct(.data$iterNG, .data$iterPar) %>%
          nrow()
        num_coef0_mod <- ifelse(num_coef0_mod > iterations, iterations, num_coef0_mod)
        if (!quiet) {
          message(paste(
            "This trial contains", num_coef0_mod, "iterations with all media coefficient = 0.",
            "Please reconsider your media variable choice if the pareto choices are unreasonable.",
            "\n   Recommendations:",
            "\n1. Increase hyperparameter ranges for 0-coef channels to give Robyn more freedom",
            "\n2. Split media into sub-channels, and/or aggregate similar channels, and/or introduce other media",
            "\n3. Increase trials to get more samples"
          ))
        }
      }
      model_output["trial"] <- ngt
      OutputModels[[ngt]] <- model_output
    }
  }
  names(OutputModels) <- paste0("trial", seq_along(OutputModels))
  return(OutputModels)
}


####################################################################
#' Core MMM Function
#'
#' \code{robyn_mmm()} function activates Nevergrad to generate samples of
#' hyperparameters, conducts media transformation within each loop, fits the
#' Ridge regression, calibrates the model optionally, decomposes responses
#' and collects the result. It's an inner function within \code{robyn_run()}.
#'
#' @inheritParams robyn_run
#' @inheritParams robyn_allocator
#' @param hyper_collect List. Containing hyperparameter bounds. Defaults to
#' \code{InputCollect$hyperparameters}.
#' @param iterations Integer. Number of iterations to run.
#' @param trial Integer. Which trial are we running? Used to ID each model.
#' @return List. MMM results with hyperparameters values.
#' @export
robyn_mmm <- function(InputCollect,
                      hyper_collect,
                      iterations,
                      cores,
                      nevergrad_algo,
                      intercept = TRUE,
                      intercept_sign,
                      ts_validation = TRUE,
                      add_penalty_factor = FALSE,
                      dt_hyper_fixed = NULL,
                      # lambda_fixed = NULL,
                      rssd_zero_penalty = TRUE,
                      refresh = FALSE,
                      trial = 1L,
                      seed = 123L,
                      quiet = FALSE, ...) {
  if (reticulate::py_module_available("nevergrad")) {
    ng <- reticulate::import("nevergrad", delay_load = TRUE)
    if (is.integer(seed)) {
      np <- reticulate::import("numpy", delay_load = FALSE)
      np$random$seed(seed)
    }
  } else {
    stop(
      "You must have nevergrad python library installed.\nPlease check our install demo: ",
      "https://github.com/facebookexperimental/Robyn/blob/main/demo/install_nevergrad.R"
    )
  }

  ################################################
  #### Collect hyperparameters

  if (TRUE) {
    hypParamSamName <- names(hyper_collect$hyper_list_all)
    # Optimization hyper-parameters
    hyper_bound_list_updated <- hyper_collect$hyper_bound_list_updated
    hyper_bound_list_updated_name <- names(hyper_bound_list_updated)
    hyper_count <- length(hyper_bound_list_updated_name)
    # Fixed hyper-parameters
    hyper_bound_list_fixed <- hyper_collect$hyper_bound_list_fixed
    hyper_bound_list_fixed_name <- names(hyper_bound_list_fixed)
    hyper_count_fixed <- length(hyper_bound_list_fixed_name)
    dt_hyper_fixed_mod <- hyper_collect$dt_hyper_fixed_mod
    hyper_fixed <- hyper_collect$all_fixed
  }

  ################################################
  #### Setup environment

  if (is.null(InputCollect$dt_mod)) {
    stop("Run InputCollect$dt_mod <- robyn_engineering() first to get the dt_mod")
  }

  ## Get environment for parallel backend
  if (TRUE) {
    dt_mod <- InputCollect$dt_mod
    xDecompAggPrev <- InputCollect$xDecompAggPrev
    rollingWindowStartWhich <- InputCollect$rollingWindowStartWhich
    rollingWindowEndWhich <- InputCollect$rollingWindowEndWhich
    refreshAddedStart <- InputCollect$refreshAddedStart
    dt_modRollWind <- InputCollect$dt_modRollWind
    refresh_steps <- InputCollect$refresh_steps
    rollingWindowLength <- InputCollect$rollingWindowLength
    paid_media_spends <- InputCollect$paid_media_spends
    organic_vars <- InputCollect$organic_vars
    context_vars <- InputCollect$context_vars
    prophet_vars <- InputCollect$prophet_vars
    adstock <- InputCollect$adstock
    context_signs <- InputCollect$context_signs
    paid_media_signs <- InputCollect$paid_media_signs
    prophet_signs <- InputCollect$prophet_signs
    organic_signs <- InputCollect$organic_signs
    calibration_input <- InputCollect$calibration_input
    optimizer_name <- nevergrad_algo
    i <- NULL # For parallel iterations (globalVar)
  }

  ################################################
  #### Get spend share

  dt_inputTrain <- InputCollect$dt_input[rollingWindowStartWhich:rollingWindowEndWhich, ]
  temp <- select(dt_inputTrain, all_of(paid_media_spends))
  dt_spendShare <- data.frame(
    rn = paid_media_spends,
    total_spend = unlist(summarise_all(temp, sum)),
    # mean_spend = unlist(summarise_all(temp, function(x) {
    #   ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0]))
    # }))
    mean_spend = unlist(summarise_all(temp, mean))
  ) %>%
    mutate(spend_share = .data$total_spend / sum(.data$total_spend))
  # When not refreshing, dt_spendShareRF = dt_spendShare
  refreshAddedStartWhich <- which(dt_modRollWind$ds == refreshAddedStart)
  temp <- select(dt_inputTrain, all_of(paid_media_spends)) %>%
    slice(refreshAddedStartWhich:rollingWindowLength)
  dt_spendShareRF <- data.frame(
    rn = paid_media_spends,
    total_spend = unlist(summarise_all(temp, sum)),
    # mean_spend = unlist(summarise_all(temp, function(x) {
    #   ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0]))
    # }))
    mean_spend = unlist(summarise_all(temp, mean))
  ) %>%
    mutate(spend_share = .data$total_spend / sum(.data$total_spend))
  # Join both dataframes into a single one
  dt_spendShare <- left_join(dt_spendShare, dt_spendShareRF, "rn", suffix = c("", "_refresh"))

  ################################################
  #### Get lambda
  lambda_min_ratio <- 0.0001 # default  value from glmnet
  lambdas <- lambda_seq(
    x = select(dt_mod, -.data$ds, -.data$dep_var),
    y = dt_mod$dep_var,
    seq_len = 100, lambda_min_ratio
  )
  lambda_max <- max(lambdas) * 0.1
  lambda_min <- lambda_max * lambda_min_ratio

  ################################################
  #### Start Nevergrad loop
  t0 <- Sys.time()

  ## Set iterations
  # hyper_fixed <- hyper_count == 0
  if (hyper_fixed == FALSE) {
    iterTotal <- iterations
    iterPar <- cores
    iterNG <- ceiling(iterations / cores) # Sometimes the progress bar may not get to 100%
  } else {
    iterTotal <- iterPar <- iterNG <- 1
  }

  ## Start Nevergrad optimizer
  if (!hyper_fixed) {
    my_tuple <- tuple(hyper_count)
    instrumentation <- ng$p$Array(shape = my_tuple, lower = 0, upper = 1)
    optimizer <- ng$optimizers$registry[optimizer_name](instrumentation, budget = iterTotal, num_workers = cores)
    # Set multi-objective dimensions for objective functions (errors)
    if (is.null(calibration_input)) {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1, 1))
    } else {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1, 1, 1))
    }
  }

  ## Prepare loop
  resultCollectNG <- list()
  cnt <- 0
  if (!hyper_fixed && !quiet) pb <- txtProgressBar(max = iterTotal, style = 3)

  sysTimeDopar <- tryCatch(
    {
      system.time({
        for (lng in 1:iterNG) { # lng = 1
          nevergrad_hp <- list()
          nevergrad_hp_val <- list()
          hypParamSamList <- list()
          hypParamSamNG <- NULL

          if (hyper_fixed == FALSE) {
            # Setting initial seeds (co = cores)
            for (co in 1:iterPar) { # co = 1
              ## Get hyperparameter sample with ask (random)
              nevergrad_hp[[co]] <- optimizer$ask()
              nevergrad_hp_val[[co]] <- nevergrad_hp[[co]]$value
              ## Scale sample to given bounds using uniform distribution
              for (hypNameLoop in hyper_bound_list_updated_name) {
                index <- which(hypNameLoop == hyper_bound_list_updated_name)
                channelBound <- unlist(hyper_bound_list_updated[hypNameLoop])
                hyppar_value <- signif(nevergrad_hp_val[[co]][index], 6)
                if (length(channelBound) > 1) {
                  hypParamSamNG[hypNameLoop] <- qunif(hyppar_value, min(channelBound), max(channelBound))
                } else {
                  hypParamSamNG[hypNameLoop] <- hyppar_value
                }
              }
              hypParamSamList[[co]] <- data.frame(t(hypParamSamNG))
            }
            hypParamSamNG <- bind_rows(hypParamSamList)
            names(hypParamSamNG) <- hyper_bound_list_updated_name
            ## Add fixed hyperparameters
            if (hyper_count_fixed != 0) {
              hypParamSamNG <- cbind(hypParamSamNG, dt_hyper_fixed_mod) %>%
                select(all_of(hypParamSamName))
            }
          } else {
            hypParamSamNG <- select(dt_hyper_fixed_mod, all_of(hypParamSamName))
          }

          # Must remain within this function for it to work
          robyn_iterations <- function(i, ...) { # i=1
            t1 <- Sys.time()
            #### Get hyperparameter sample
            hypParamSam <- hypParamSamNG[i, ]
            adstock <- check_adstock(adstock)

            #### Transform media for model fitting
            temp <- run_transformations(InputCollect, hypParamSam, adstock)
            dt_modSaturated <- temp$dt_modSaturated
            dt_saturatedImmediate <- temp$dt_saturatedImmediate
            dt_saturatedCarryover <- temp$dt_saturatedCarryover

            #####################################
            #### Split train & test and prepare data for modelling

            dt_window <- dt_modSaturated

            ## Contrast matrix because glmnet does not treat categorical variables (one hot encoding)
            y_window <- dt_window$dep_var
            x_window <- as.matrix(lares::ohse(select(dt_window, -.data$dep_var)))
            y_train <- y_val <- y_test <- y_window
            x_train <- x_val <- x_test <- x_window

            ## Split train, test, and validation sets
            train_size <- hypParamSam[, "train_size"][[1]]
            val_size <- test_size <- (1 - train_size) / 2
            if (train_size < 1) {
              train_size_index <- floor(quantile(seq(nrow(dt_window)), train_size))
              val_size_index <- train_size_index + floor(val_size * nrow(dt_window))
              y_train <- y_window[1:train_size_index]
              y_val <- y_window[(train_size_index + 1):val_size_index]
              y_test <- y_window[(val_size_index + 1):length(y_window)]
              x_train <- x_window[1:train_size_index, ]
              x_val <- x_window[(train_size_index + 1):val_size_index, ]
              x_test <- x_window[(val_size_index + 1):length(y_window), ]
            } else {
              y_val <- y_test <- x_val <- x_test <- NULL
            }

            ## Define and set sign control
            dt_sign <- select(dt_window, -.data$dep_var)
            x_sign <- c(prophet_signs, context_signs, paid_media_signs, organic_signs)
            names(x_sign) <- c(prophet_vars, context_vars, paid_media_spends, organic_vars)
            check_factor <- unlist(lapply(dt_sign, is.factor))
            lower.limits <- rep(0, length(prophet_signs))
            upper.limits <- rep(1, length(prophet_signs))
            for (s in (length(prophet_signs) + 1):length(x_sign)) {
              if (check_factor[s] == TRUE) {
                level.n <- length(levels(unlist(dt_sign[, s, with = FALSE])))
                if (level.n <= 1) {
                  stop("All factor variables must have more than 1 level")
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
            #### Fit ridge regression with nevergrad's lambda
            # lambdas <- lambda_seq(x_train, y_train, seq_len = 100, lambda_min_ratio = 0.0001)
            # lambda_max <- max(lambdas)
            lambda_hp <- unlist(hypParamSamNG$lambda[i])
            if (hyper_fixed == FALSE) {
              lambda_scaled <- lambda_min + (lambda_max - lambda_min) * lambda_hp
            } else {
              lambda_scaled <- lambda_hp
            }

            if (add_penalty_factor) {
              penalty.factor <- unlist(hypParamSamNG[i, grepl("_penalty", names(hypParamSamNG))])
            } else {
              penalty.factor <- rep(1, ncol(x_train))
            }

            #####################################
            ## NRMSE: Model's fit error

            ## If no lift calibration, refit using best lambda
            mod_out <- model_refit(
              x_train, y_train,
              x_val, y_val,
              x_test, y_test,
              lambda = lambda_scaled,
              lower.limits = lower.limits,
              upper.limits = upper.limits,
              intercept = intercept,
              intercept_sign = intercept_sign,
              penalty.factor = penalty.factor,
              ...
            )
            decompCollect <- model_decomp(
              coefs = mod_out$coefs,
              y_pred = mod_out$y_pred,
              dt_modSaturated = dt_modSaturated,
              dt_saturatedImmediate = dt_saturatedImmediate,
              dt_saturatedCarryover = dt_saturatedCarryover,
              dt_modRollWind = dt_modRollWind,
              refreshAddedStart = refreshAddedStart
            )
            nrmse <- ifelse(ts_validation, mod_out$nrmse_val, mod_out$nrmse_train)
            mape <- 0
            df.int <- mod_out$df.int

            #####################################
            #### MAPE: Calibration error
            if (!is.null(calibration_input)) {
              liftCollect <- robyn_calibrate(
                calibration_input = calibration_input,
                df_raw = dt_mod,
                hypParamSam = hypParamSam,
                wind_start = rollingWindowStartWhich,
                wind_end = rollingWindowEndWhich,
                dayInterval = InputCollect$dayInterval,
                adstock = adstock,
                xDecompVec = decompCollect$xDecompVec,
                coefs = decompCollect$coefsOutCat
              )
              mape <- mean(liftCollect$mape_lift, na.rm = TRUE)
            }

            #####################################
            #### DECOMP.RSSD: Business error
            # Sum of squared distance between decomp share and spend share to be minimized
            dt_decompSpendDist <- decompCollect$xDecompAgg %>%
              filter(.data$rn %in% paid_media_spends) %>%
              select(
                .data$rn, .data$xDecompAgg, .data$xDecompPerc, .data$xDecompMeanNon0Perc,
                .data$xDecompMeanNon0, .data$xDecompPercRF, .data$xDecompMeanNon0PercRF,
                .data$xDecompMeanNon0RF
              ) %>%
              left_join(
                select(
                  dt_spendShare,
                  .data$rn, .data$spend_share, .data$spend_share_refresh,
                  .data$mean_spend, .data$total_spend
                ),
                by = "rn"
              ) %>%
              mutate(
                effect_share = .data$xDecompPerc / sum(.data$xDecompPerc),
                effect_share_refresh = .data$xDecompPercRF / sum(.data$xDecompPercRF)
              )
            dt_decompSpendDist <- left_join(
              filter(decompCollect$xDecompAgg, .data$rn %in% paid_media_spends),
              select(dt_decompSpendDist, .data$rn, contains("_spend"), contains("_share")),
              by = "rn"
            )
            if (!refresh) {
              decomp.rssd <- sqrt(sum((dt_decompSpendDist$effect_share - dt_decompSpendDist$spend_share)^2))
              # Penalty for models with more 0-coefficients
              if (rssd_zero_penalty) {
                is_0eff <- round(dt_decompSpendDist$effect_share, 4) == 0
                share_0eff <- sum(is_0eff) / length(dt_decompSpendDist$effect_share)
                decomp.rssd <- decomp.rssd * (1 + share_0eff)
              }
            } else {
              dt_decompRF <- select(decompCollect$xDecompAgg, .data$rn, decomp_perc = .data$xDecompPerc) %>%
                left_join(select(xDecompAggPrev, .data$rn, decomp_perc_prev = .data$xDecompPerc),
                  by = "rn"
                )
              decomp.rssd.media <- dt_decompRF %>%
                filter(.data$rn %in% paid_media_spends) %>%
                summarise(rssd.media = sqrt(mean((.data$decomp_perc - .data$decomp_perc_prev)^2))) %>%
                pull(.data$rssd.media)
              decomp.rssd.nonmedia <- dt_decompRF %>%
                filter(!.data$rn %in% paid_media_spends) %>%
                summarise(rssd.nonmedia = sqrt(mean((.data$decomp_perc - .data$decomp_perc_prev)^2))) %>%
                pull(.data$rssd.nonmedia)
              decomp.rssd <- decomp.rssd.media + decomp.rssd.nonmedia /
                (1 - refresh_steps / rollingWindowLength)
            }
            # When all media in this iteration have 0 coefficients
            if (is.nan(decomp.rssd)) {
              decomp.rssd <- Inf
              dt_decompSpendDist$effect_share <- 0
            }

            #####################################
            #### Collect Multi-Objective Errors and Iteration Results
            resultCollect <- list()

            # Auxiliary dynamic vector
            common <- data.frame(
              rsq_train = mod_out$rsq_train,
              rsq_val = mod_out$rsq_val,
              rsq_test = mod_out$rsq_test,
              nrmse_train = mod_out$nrmse_train,
              nrmse_val = mod_out$nrmse_val,
              nrmse_test = mod_out$nrmse_test,
              nrmse = nrmse,
              decomp.rssd = decomp.rssd,
              mape = mape,
              lambda = lambda_scaled,
              lambda_hp = lambda_hp,
              lambda_max = lambda_max,
              lambda_min_ratio = lambda_min_ratio,
              solID = paste(trial, lng, i, sep = "_"),
              trial = trial,
              iterNG = lng,
              iterPar = i
            )

            total_common <- ncol(common)
            split_common <- which(colnames(common) == "lambda_min_ratio")

            resultCollect[["resultHypParam"]] <- as_tibble(hypParamSam) %>%
              select(-.data$lambda) %>%
              bind_cols(common[, 1:split_common]) %>%
              mutate(
                pos = prod(decompCollect$xDecompAgg$pos),
                Elapsed = as.numeric(difftime(Sys.time(), t1, units = "secs")),
                ElapsedAccum = as.numeric(difftime(Sys.time(), t0, units = "secs"))
              ) %>%
              bind_cols(common[, (split_common + 1):total_common]) %>%
              dplyr::mutate_all(unlist)

            resultCollect[["xDecompAgg"]] <- decompCollect$xDecompAgg %>%
              mutate(train_size = train_size) %>%
              bind_cols(common)

            if (!is.null(calibration_input)) {
              resultCollect[["liftCalibration"]] <- liftCollect %>%
                bind_cols(common)
            }

            resultCollect[["decompSpendDist"]] <- dt_decompSpendDist %>%
              bind_cols(common)

            resultCollect <- append(resultCollect, as.list(common))
            return(resultCollect)
          }

          ########### Parallel start
          nrmse.collect <- NULL
          decomp.rssd.collect <- NULL
          best_mape <- Inf
          if (cores == 1) {
            doparCollect <- lapply(1:iterPar, robyn_iterations)
          } else {
            # Create cluster to minimize overhead for parallel back-end registering
            if (check_parallel() && !hyper_fixed) {
              registerDoParallel(cores)
            } else {
              registerDoSEQ()
            }
            suppressPackageStartupMessages(
              doparCollect <- foreach(i = 1:iterPar) %dorng% robyn_iterations(i)
            )
          }

          nrmse.collect <- unlist(lapply(doparCollect, function(x) x$nrmse))
          decomp.rssd.collect <- unlist(lapply(doparCollect, function(x) x$decomp.rssd))
          mape.lift.collect <- unlist(lapply(doparCollect, function(x) x$mape))

          #####################################
          #### Nevergrad tells objectives

          if (!hyper_fixed) {
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
          if (!quiet) {
            cnt <- cnt + iterPar
            if (!hyper_fixed) setTxtProgressBar(pb, cnt)
          }
        } ## end NG loop
      }) # end system.time
    },
    error = function(err) {
      if (length(resultCollectNG) > 1) {
        msg <- "Error while running robyn_mmm(); providing PARTIAL results"
        warning(msg)
        message(paste(msg, err, sep = "\n"))
        sysTimeDopar <- rep(Sys.time() - t0, 3)
      } else {
        stop(err)
      }
    }
  )

  # stop cluster to avoid memory leaks
  stopImplicitCluster()
  registerDoSEQ()
  getDoParWorkers()

  if (!hyper_fixed) {
    cat("\r", paste("\n  Finished in", round(sysTimeDopar[3] / 60, 2), "mins"))
    flush.console()
    close(pb)
  }

  #####################################
  #### Final result collect

  resultCollect <- list()

  resultCollect[["resultHypParam"]] <- as_tibble(bind_rows(
    lapply(resultCollectNG, function(x) {
      bind_rows(lapply(x, function(y) y$resultHypParam))
    })
  ))

  # resultCollect[["xDecompVec"]] <- as_tibble(bind_rows(
  #   lapply(resultCollectNG, function(x) {
  #     bind_rows(lapply(x, function(y) y$xDecompVec))
  #   })
  # ))

  resultCollect[["xDecompAgg"]] <- as_tibble(bind_rows(
    lapply(resultCollectNG, function(x) {
      bind_rows(lapply(x, function(y) y$xDecompAgg))
    })
  ))

  if (!is.null(calibration_input)) {
    resultCollect[["liftCalibration"]] <- as_tibble(bind_rows(
      lapply(resultCollectNG, function(x) {
        bind_rows(lapply(x, function(y) y$liftCalibration))
      })
    ) %>%
      arrange(.data$mape, .data$liftMedia, .data$liftStart))
  }

  resultCollect[["decompSpendDist"]] <- as_tibble(bind_rows(
    lapply(resultCollectNG, function(x) {
      bind_rows(lapply(x, function(y) y$decompSpendDist))
    })
  ))

  resultCollect$iter <- length(resultCollect$mape)
  resultCollect$elapsed.min <- sysTimeDopar[3] / 60

  # Adjust accumulated time
  resultCollect$resultHypParam <- resultCollect$resultHypParam %>%
    mutate(ElapsedAccum = .data$ElapsedAccum - min(.data$ElapsedAccum) +
      .data$Elapsed[which.min(.data$ElapsedAccum)])

  return(list(
    resultCollect = resultCollect,
    hyperBoundNG = hyper_bound_list_updated,
    hyperBoundFixed = hyper_bound_list_fixed
  ))
}

model_decomp <- function(coefs, y_pred,
                         dt_modSaturated, dt_saturatedImmediate,
                         dt_saturatedCarryover, dt_modRollWind,
                         refreshAddedStart) {
  ## Input for decomp
  y <- dt_modSaturated$dep_var
  # x <- data.frame(x)

  x <- select(dt_modSaturated, -.data$dep_var)
  intercept <- coefs[1]
  x_name <- names(x)
  x_factor <- x_name[sapply(x, is.factor)]

  ## Decomp x
  xDecomp <- data.frame(mapply(function(regressor, coeff) {
    regressor * coeff
  }, regressor = x, coeff = coefs[-1]))
  xDecomp <- cbind(data.frame(intercept = rep(intercept, nrow(xDecomp))), xDecomp)
  xDecompOut <- cbind(data.frame(ds = dt_modRollWind$ds, y = y, y_pred = y_pred), xDecomp)

  ## Decomp immediate & carryover response
  sel_coef <- c(rownames(coefs), names(coefs)) %in% names(dt_saturatedImmediate)
  coefs_media <- coefs[sel_coef]
  names(coefs_media) <- rownames(coefs)[sel_coef]
  mediaDecompImmediate <- data.frame(mapply(function(regressor, coeff) {
    regressor * coeff
  }, regressor = dt_saturatedImmediate, coeff = coefs_media))
  mediaDecompCarryover <- data.frame(mapply(function(regressor, coeff) {
    regressor * coeff
  }, regressor = dt_saturatedCarryover, coeff = coefs_media))

  ## Output decomp
  y_hat <- rowSums(xDecomp, na.rm = TRUE)
  y_hat.scaled <- rowSums(abs(xDecomp), na.rm = TRUE)
  xDecompOutPerc.scaled <- abs(xDecomp) / y_hat.scaled
  xDecompOut.scaled <- y_hat * xDecompOutPerc.scaled

  temp <- select(xDecompOut, .data$intercept, all_of(x_name))
  xDecompOutAgg <- sapply(temp, function(x) sum(x))
  xDecompOutAggPerc <- xDecompOutAgg / sum(y_hat)
  xDecompOutAggMeanNon0 <- unlist(lapply(temp, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x != 0]))))
  xDecompOutAggMeanNon0[is.nan(xDecompOutAggMeanNon0)] <- 0
  xDecompOutAggMeanNon0Perc <- xDecompOutAggMeanNon0 / sum(xDecompOutAggMeanNon0)

  refreshAddedStartWhich <- which(xDecompOut$ds == refreshAddedStart)
  refreshAddedEnd <- max(xDecompOut$ds)
  refreshAddedEndWhich <- which(xDecompOut$ds == refreshAddedEnd)

  temp <- select(xDecompOut, .data$intercept, all_of(x_name)) %>%
    slice(refreshAddedStartWhich:refreshAddedEndWhich)
  xDecompOutAggRF <- unlist(lapply(temp, function(x) sum(x)))
  y_hatRF <- y_hat[refreshAddedStartWhich:refreshAddedEndWhich]
  xDecompOutAggPercRF <- xDecompOutAggRF / sum(y_hatRF)
  xDecompOutAggMeanNon0RF <- unlist(lapply(temp, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x != 0]))))
  xDecompOutAggMeanNon0RF[is.nan(xDecompOutAggMeanNon0RF)] <- 0
  xDecompOutAggMeanNon0PercRF <- xDecompOutAggMeanNon0RF / sum(xDecompOutAggMeanNon0RF)

  coefsOutCat <- coefsOut <- data.frame(rn = c(rownames(coefs), names(coefs)), coefs)
  if (length(x_factor) > 0) {
    coefsOut$rn <- sapply(x_factor, function(x) str_replace(coefsOut$rn, paste0(x, ".*"), x))
  }
  rn_order <- names(xDecompOutAgg)
  rn_order[rn_order == "intercept"] <- "(Intercept)"
  coefsOut <- coefsOut %>%
    group_by(.data$rn) %>%
    rename("coef" = 2) %>%
    summarise(coef = mean(.data$coef)) %>%
    arrange(match(.data$rn, rn_order))

  decompOutAgg <- as_tibble(cbind(coefsOut, data.frame(
    xDecompAgg = xDecompOutAgg,
    xDecompPerc = xDecompOutAggPerc,
    xDecompMeanNon0 = xDecompOutAggMeanNon0,
    xDecompMeanNon0Perc = xDecompOutAggMeanNon0Perc,
    xDecompAggRF = xDecompOutAggRF,
    xDecompPercRF = xDecompOutAggPercRF,
    xDecompMeanNon0RF = xDecompOutAggMeanNon0RF,
    xDecompMeanNon0PercRF = xDecompOutAggMeanNon0PercRF,
    pos = xDecompOutAgg >= 0
  )))

  decompCollect <- list(
    xDecompVec = xDecompOut, xDecompVec.scaled = xDecompOut.scaled,
    xDecompAgg = decompOutAgg, coefsOutCat = coefsOutCat,
    mediaDecompImmediate = mutate(mediaDecompImmediate, ds = xDecompOut$ds, y = xDecompOut$y),
    mediaDecompCarryover = mutate(mediaDecompCarryover, ds = xDecompOut$ds, y = xDecompOut$y)
  )
  return(decompCollect)
}

model_refit <- function(x_train, y_train, x_val, y_val, x_test, y_test,
                        lambda, lower.limits, upper.limits,
                        intercept = TRUE,
                        intercept_sign = "non_negative",
                        penalty.factor = rep(1, ncol(y_train)),
                        ...) {
  mod <- glmnet(
    x_train,
    y_train,
    family = "gaussian",
    alpha = 0, # 0 for ridge regression
    lambda = lambda,
    lower.limits = lower.limits,
    upper.limits = upper.limits,
    type.measure = "mse",
    penalty.factor = penalty.factor,
    intercept = intercept,
    ...
  ) # coef(mod)

  df.int <- 1

  ## Drop intercept if negative and intercept_sign == "non_negative"
  if (intercept_sign == "non_negative" && coef(mod)[1] < 0) {
    mod <- glmnet(
      x_train,
      y_train,
      family = "gaussian",
      alpha = 0, # 0 for ridge regression
      lambda = lambda,
      lower.limits = lower.limits,
      upper.limits = upper.limits,
      penalty.factor = penalty.factor,
      intercept = FALSE,
      ...
    ) # coef(mod)
    df.int <- 0
  } # plot(mod); print(mod)

  # Calculate all Adjusted R2
  y_train_pred <- as.vector(predict(mod, s = lambda, newx = x_train))
  rsq_train <- get_rsq(true = y_train, predicted = y_train_pred, p = ncol(x_train), df.int = df.int)
  if (!is.null(x_val)) {
    y_val_pred <- as.vector(predict(mod, s = lambda, newx = x_val))
    rsq_val <- get_rsq(true = y_val, predicted = y_val_pred, p = ncol(x_val), df.int = df.int, n_train = length(y_train))
    y_test_pred <- as.vector(predict(mod, s = lambda, newx = x_test))
    rsq_test <- get_rsq(true = y_test, predicted = y_test_pred, p = ncol(x_test), df.int = df.int, n_train = length(y_train))
    y_pred <- c(y_train_pred, y_val_pred, y_test_pred)
  } else {
    rsq_val <- rsq_test <- NA
    y_pred <- y_train_pred
  }

  # Calculate all NRMSE
  nrmse_train <- sqrt(mean((y_train - y_train_pred)^2)) / (max(y_train) - min(y_train))
  if (!is.null(x_val)) {
    nrmse_val <- sqrt(mean(sum((y_val - y_val_pred)^2))) / (max(y_val) - min(y_val))
    nrmse_test <- sqrt(mean(sum((y_test - y_test_pred)^2))) / (max(y_test) - min(y_test))
  } else {
    nrmse_val <- nrmse_test <- y_val_pred <- y_test_pred <- NA
  }

  mod_out <- list(
    rsq_train = rsq_train,
    rsq_val = rsq_val,
    rsq_test = rsq_test,
    nrmse_train = nrmse_train,
    nrmse_val = nrmse_val,
    nrmse_test = nrmse_test,
    coefs = as.matrix(coef(mod)),
    y_train_pred = y_train_pred,
    y_val_pred = y_val_pred,
    y_test_pred = y_test_pred,
    y_pred = y_pred,
    mod = mod,
    df.int = df.int
  )

  return(mod_out)
}

# x = x_train matrix
# y = y_train (dep_var) vector
lambda_seq <- function(x, y, seq_len = 100, lambda_min_ratio = 0.0001) {
  mysd <- function(y) sqrt(sum((y - mean(y))^2) / length(y))
  sx <- scale(x, scale = apply(x, 2, mysd))
  check_nan <- apply(sx, 2, function(sxj) all(is.nan(sxj)))
  sx <- mapply(function(sxj, v) {
    return(if (v) rep(0, length(sxj)) else sxj)
  }, sxj = as.data.frame(sx), v = check_nan)
  sx <- as.matrix(sx, ncol = ncol(x), nrow = nrow(x))
  # sy <- as.vector(scale(y, scale=mysd(y)))
  sy <- y
  # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)
  lambda_max <- max(abs(colSums(sx * sy))) / (0.001 * nrow(x))
  lambda_max_log <- log(lambda_max)
  log_step <- (log(lambda_max) - log(lambda_max * lambda_min_ratio)) / (seq_len - 1)
  log_seq <- seq(log(lambda_max), log(lambda_max * lambda_min_ratio), length.out = seq_len)
  lambdas <- exp(log_seq)
  return(lambdas)
}

hyper_collector <- function(InputCollect, hyper_in, ts_validation, add_penalty_factor, dt_hyper_fixed = NULL, cores) {
  # Fetch hyper-parameters based on media
  hypParamSamName <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)

  # Manually add other hyper-parameters
  hypParamSamName <- c(hypParamSamName, HYPS_OTHERS)

  # Add penalty factor hyper-parameters names
  for_penalty <- names(select(InputCollect$dt_mod, -.data$ds, -.data$dep_var))
  if (add_penalty_factor) hypParamSamName <- c(hypParamSamName, paste0("penalty_", for_penalty))

  # Check hyper_fixed condition + add lambda + penalty factor hyper-parameters names
  all_fixed <- check_hyper_fixed(InputCollect, dt_hyper_fixed, add_penalty_factor)
  hypParamSamName <- attr(all_fixed, "hypParamSamName")

  if (!all_fixed) {
    # Collect media hyperparameters
    hyper_bound_list <- list()
    for (i in seq_along(hypParamSamName)) {
      hyper_bound_list[i] <- hyper_in[hypParamSamName[i]]
      names(hyper_bound_list)[i] <- hypParamSamName[i]
    }

    # Add unfixed lambda hyperparameter manually
    if (length(hyper_bound_list[["lambda"]]) != 1) {
      hyper_bound_list$lambda <- c(0, 1)
    }

    # Add unfixed train_size hyperparameter manually
    if (ts_validation) {
      if (!"train_size" %in% names(hyper_bound_list)) {
        hyper_bound_list$train_size <- c(0.5, 0.8)
      }
      message(sprintf(
        "Time-series validation with train_size range of %s of the data...",
        paste(formatNum(100 * hyper_bound_list$train_size, pos = "%"), collapse = "-")
      ))
    } else {
      if ("train_size" %in% names(hyper_bound_list)) {
        warning("Provided train_size but ts_validation = FALSE. Time series validation inactive.")
      }
      hyper_bound_list$train_size <- 1
      message("Fitting time series with all available data...")
    }

    # Add unfixed penalty.factor hyperparameters manually
    for_penalty <- names(select(InputCollect$dt_mod, -.data$ds, -.data$dep_var))
    penalty_names <- paste0(for_penalty, "_penalty")
    if (add_penalty_factor) {
      for (penalty in penalty_names) {
        if (length(hyper_bound_list[[penalty]]) != 1) {
          hyper_bound_list[[penalty]] <- c(0, 1)
        }
      }
    }

    # Get hyperparameters for Nevergrad
    hyper_bound_list_updated <- hyper_bound_list[which(unlist(lapply(hyper_bound_list, length) == 2))]

    # Get fixed hyperparameters
    hyper_bound_list_fixed <- hyper_bound_list[which(unlist(lapply(hyper_bound_list, length) == 1))]

    hyper_list_bind <- c(hyper_bound_list_updated, hyper_bound_list_fixed)
    hyper_list_all <- list()
    for (i in seq_along(hypParamSamName)) {
      hyper_list_all[[i]] <- hyper_list_bind[[hypParamSamName[i]]]
      names(hyper_list_all)[i] <- hypParamSamName[i]
    }

    dt_hyper_fixed_mod <- data.frame(bind_cols(lapply(hyper_bound_list_fixed, function(x) rep(x, cores))))
  } else {
    hyper_bound_list_fixed <- list()
    for (i in seq_along(hypParamSamName)) {
      hyper_bound_list_fixed[[i]] <- dt_hyper_fixed[[hypParamSamName[i]]]
      names(hyper_bound_list_fixed)[i] <- hypParamSamName[i]
    }

    hyper_list_all <- hyper_bound_list_fixed
    hyper_bound_list_updated <- hyper_bound_list_fixed[which(unlist(lapply(hyper_bound_list_fixed, length) == 2))]
    cores <- 1

    dt_hyper_fixed_mod <- data.frame(matrix(hyper_bound_list_fixed, nrow = 1))
    names(dt_hyper_fixed_mod) <- names(hyper_bound_list_fixed)
  }

  return(list(
    hyper_list_all = hyper_list_all,
    hyper_bound_list_updated = hyper_bound_list_updated,
    hyper_bound_list_fixed = hyper_bound_list_fixed,
    dt_hyper_fixed_mod = dt_hyper_fixed_mod,
    all_fixed = all_fixed
  ))
}

init_msgs_run <- function(InputCollect, refresh, lambda_control = NULL, quiet = FALSE) {
  if (!is.null(lambda_control)) {
    message("Input 'lambda_control' deprecated in v3.6.0; lambda is now selected by hyperparameter optimization")
  }
  if (!quiet) {
    message(sprintf(
      "Input data has %s %ss in total: %s to %s",
      nrow(InputCollect$dt_mod),
      InputCollect$intervalType,
      min(InputCollect$dt_mod$ds),
      max(InputCollect$dt_mod$ds)
    ))
    depth <- ifelse(
      "refreshDepth" %in% names(InputCollect),
      InputCollect$refreshDepth,
      ifelse("refreshCounter" %in% names(InputCollect),
        InputCollect$refreshCounter, 0
      )
    )
    refresh <- as.integer(depth) > 0
    message(sprintf(
      "%s model is built on rolling window of %s %s: %s to %s",
      ifelse(!refresh, "Initial", paste0("Refresh #", depth)),
      InputCollect$rollingWindowLength,
      InputCollect$intervalType,
      InputCollect$window_start,
      InputCollect$window_end
    ))
    if (refresh) {
      message(sprintf(
        "Rolling window moving forward: %s %ss",
        InputCollect$refresh_steps, InputCollect$intervalType
      ))
    }
  }
}
