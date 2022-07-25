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
#' @param dt_hyper_fixed data.frame. Only provide when loading old model results.
#' It consumes hyperparameters from saved csv \code{pareto_hyperparameters.csv}.
#' @param add_penalty_factor Boolean. Add penalty factor hyperparameters to
#' glmnet's penalty.factor to be optimized by nevergrad. Use with caution, because
#' this feature might add too much hyperparameter space and probably requires
#' more iterations to converge.
#' @param refresh Boolean. Set to \code{TRUE} when used in \code{robyn_refresh()}.
#' @param cores Integer. Default to \code{parallel::detectCores()} (max cores).
#' @param iterations Integer. Recommended 2000 for default when using
#' \code{nevergrad_algo = "TwoPointsDE"}.
#' @param trials Integer. Recommended 5 for default
#' \code{nevergrad_algo = "TwoPointsDE"}.
#' @param nevergrad_algo Character. Default to "TwoPointsDE". Options are
#' \code{c("DE","TwoPointsDE", "OnePlusOne", "DoubleFastGADiscreteOnePlusOne",
#' "DiscreteOnePlusOne", "PortfolioDiscreteOnePlusOne", "NaiveTBPSA",
#' "cGA", "RandomSearch")}.
#' @param intercept_sign Character. Choose one of "non_negative" (default) or
#' "unconstrained". By default, if intercept is negative, Robyn will drop intercept
#' and refit the model. Consider changing intercept_sign to "unconstrained" when
#' there are \code{context_vars} with large positive values.
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
robyn_run <- function(InputCollect,
                      dt_hyper_fixed = NULL,
                      add_penalty_factor = FALSE,
                      refresh = FALSE,
                      seed = 123L,
                      outputs = FALSE,
                      quiet = FALSE,
                      cores = NULL,
                      trials = NULL,
                      iterations = NULL,
                      nevergrad_algo = "TwoPointsDE",
                      intercept_sign = "non_negative",
                      lambda_control = NULL,
                      ...) {
  t0 <- Sys.time()

  #####################################
  #### Set local environment

  if (!"hyperparameters" %in% names(InputCollect) | is.null(InputCollect$hyperparameters)) {
    stop("Must provide 'hyperparameters' in robyn_inputs()'s output first")
  }

  # Check and warn on legacy inputs (using InputCollect params as robyn_run() inputs)
  InputCollect <- check_legacy_input(InputCollect, cores, iterations, trials, intercept_sign, nevergrad_algo)
  # Overwrite values imported from InputCollect
  legacyValues <- InputCollect[LEGACY_PARAMS]
  legacyValues <- legacyValues[!sapply(legacyValues, is.null)]
  if (length(legacyValues) > 0) {
    for (i in 1:length(InputCollect)) assign(names(InputCollect)[i], InputCollect[[i]])
  }

  if (is.null(cores)) cores <- parallel::detectCores()
  hyps_fixed <- !is.null(dt_hyper_fixed)
  if (hyps_fixed) trials <- iterations <- 1
  check_run_inputs(cores, iterations, trials, intercept_sign, nevergrad_algo)
  check_iteration(InputCollect$calibration_input, iterations, trials, hyps_fixed, refresh)
  init_msgs_run(InputCollect, refresh, lambda_control, quiet)

  #####################################
  #### Prepare hyper-parameters

  hyper_collect <- hyper_collector(InputCollect, InputCollect$hyperparameters, add_penalty_factor, dt_hyper_fixed, cores)
  InputCollect$hyper_updated <- hyper_collect$hyper_list_all

  #####################################
  #### Run robyn_mmm on set_trials

  OutputModels <- robyn_train(
    InputCollect, hyper_collect,
    cores, iterations, trials, intercept_sign, nevergrad_algo,
    dt_hyper_fixed, add_penalty_factor,
    refresh, seed, quiet
  )

  attr(OutputModels, "hyper_fixed") <- hyper_collect$all_fixed
  attr(OutputModels, "refresh") <- refresh

  if (TRUE) {
    OutputModels$cores <- cores
    OutputModels$iterations <- iterations
    OutputModels$trials <- trials
    OutputModels$intercept_sign <- intercept_sign
    OutputModels$nevergrad_algo <- nevergrad_algo
    OutputModels$add_penalty_factor <- add_penalty_factor
    OutputModels$hyper_updated <- hyper_collect$hyper_list_all
  }

  if (!outputs) {
    output <- OutputModels
  } else if (!hyper_collect$all_fixed) {
    output <- robyn_outputs(InputCollect, OutputModels, ...)
  } else {
    output <- robyn_outputs(InputCollect, OutputModels, clusters = FALSE, ...)
  }

  # Check convergence
  if (!hyper_collect$all_fixed) output[["convergence"]] <- robyn_converge(OutputModels, ...)

  # Save hyper-parameters list
  output[["hyper_updated"]] <- hyper_collect$hyper_list_all

  # Report total timing
  attr(output, "runTime") <- round(difftime(Sys.time(), t0, units = "mins"), 2)
  if (!quiet) message(paste("Total run time:", attr(output, "runTime"), "mins"))

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
  Intercept sign: {x$intercept_sign}
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
                        intercept_sign, nevergrad_algo,
                        dt_hyper_fixed = NULL,
                        add_penalty_factor = FALSE,
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
      intercept_sign = intercept_sign,
      dt_hyper_fixed = dt_hyper_fixed,
      seed = seed,
      quiet = quiet
    )

    OutputModels[[1]]$trial <- 1
    OutputModels[[1]]$resultCollect$resultHypParam <- arrange(
      OutputModels[[1]]$resultCollect$resultHypParam, .data$iterPar)
    dt_IDs <- data.frame(
      solID = dt_hyper_fixed$solID,
      iterPar = OutputModels[[1]]$resultCollect$resultHypParam$iterPar
    )
    these <- c("resultHypParam", "xDecompAgg", "xDecompVec", "decompSpendDist")
    for (tab in these) {
      OutputModels[[1]]$resultCollect[[tab]] <- left_join(
        OutputModels[[1]]$resultCollect[[tab]], dt_IDs, by = "iterPar")
    }
  } else {

    ## Run robyn_mmm on set_trials if hyperparameters are not all fixed
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
        intercept_sign = intercept_sign,
        add_penalty_factor = add_penalty_factor,
        refresh = refresh,
        seed = seed + ngt,
        quiet = quiet
      )
      check_coef0 <- any(model_output$resultCollect$decompSpendDist$decomp.rssd == Inf)
      if (check_coef0) {
        num_coef0_mod <- filter(model_output$resultCollect$decompSpendDist, is.infinite(.data$decomp.rssd)) %>%
          distinct(.data$iterNG, .data$iterPar) %>% nrow()
        num_coef0_mod <- ifelse(num_coef0_mod > iterations, iterations, num_coef0_mod)
        if (!quiet) message(paste(
          "This trial contains", num_coef0_mod, "iterations with all media coefficient = 0.",
          "Please reconsider your media variable choice if the pareto choices are unreasonable.",
          "\n   Recommendations:",
          "\n1. Increase hyperparameter ranges for 0-coef channels to give Robyn more freedom",
          "\n2. Split media into sub-channels, and/or aggregate similar channels, and/or introduce other media",
          "\n3. Increase trials to get more samples"))
      }
      model_output["trial"] <- ngt
      OutputModels[[ngt]] <- model_output
    }
  }
  names(OutputModels) <- paste0("trial", 1:length(OutputModels))
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
#' @return List. MMM results with hyperparameters values.
#' @export
robyn_mmm <- function(InputCollect,
                      hyper_collect,
                      iterations,
                      cores,
                      nevergrad_algo,
                      intercept_sign,
                      add_penalty_factor = FALSE,
                      dt_hyper_fixed = NULL,
                      # lambda_fixed = NULL,
                      refresh = FALSE,
                      seed = 123L,
                      quiet = FALSE) {
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

  if (is.null(InputCollect$dt_mod))
    stop("Run InputCollect$dt_mod <- robyn_engineering() first to get the dt_mod")

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
    all_media <- InputCollect$all_media
    calibration_input <- InputCollect$calibration_input
    optimizer_name <- nevergrad_algo
    add_penalty_factor <- add_penalty_factor
    intercept_sign <- intercept_sign
    i <- NULL # For parallel iterations (globalVar)
  }

  ################################################
  #### Get spend share

  dt_inputTrain <- InputCollect$dt_input[rollingWindowStartWhich:rollingWindowEndWhich, ]
  temp <- select(dt_inputTrain, all_of(paid_media_spends))
  dt_spendShare <- data.frame(
    rn = paid_media_spends,
    total_spend = unlist(summarise_all(temp, sum)),
    mean_spend = unlist(summarise_all(temp, function(x) {
      ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0]))
    }))
  ) %>%
    mutate(spend_share = .data$total_spend / sum(.data$total_spend))
  # When not refreshing, dt_spendShareRF = dt_spendShare
  refreshAddedStartWhich <- which(dt_modRollWind$ds == refreshAddedStart)
  temp <- select(dt_inputTrain, all_of(paid_media_spends)) %>%
    slice(refreshAddedStartWhich:rollingWindowLength)
  dt_spendShareRF <- data.frame(
    rn = paid_media_spends,
    total_spend = unlist(summarise_all(temp, sum)),
    mean_spend = unlist(summarise_all(temp, function(x) {
      ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0]))
    }))
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
  if (!hyper_fixed & !quiet) pb <- txtProgressBar(max = iterTotal, style = 3)
  # Create cluster before big for-loop to minimize overhead for parallel back-end registering
  if (check_parallel() & !hyper_fixed) {
    registerDoParallel(cores)
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
        # Setting initial seeds
        for (co in 1:iterPar) { # co = 1
          ## Get hyperparameter sample with ask (random)
          nevergrad_hp[[co]] <- optimizer$ask()
          nevergrad_hp_val[[co]] <- nevergrad_hp[[co]]$value
          ## Scale sample to given bounds using uniform distribution
          for (hypNameLoop in hyper_bound_list_updated_name) {
            index <- which(hypNameLoop == hyper_bound_list_updated_name)
            channelBound <- unlist(hyper_bound_list_updated[hypNameLoop])
            hyppar_value <- nevergrad_hp_val[[co]][index]
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

      ## Parallel start

      nrmse.collect <- c()
      decomp.rssd.collect <- c()
      best_mape <- Inf

      doparCollect <- suppressPackageStartupMessages(
        # for (i in 1:iterPar) {
        foreach(i = 1:iterPar) %dorng% { # i = 1
          t1 <- Sys.time()
          #### Get hyperparameter sample
          hypParamSam <- hypParamSamNG[i, ]
          #### Tranform media with hyperparameters
          dt_modAdstocked <- select(dt_mod, -.data$ds)
          mediaAdstocked <- list()
          mediaVecCum <- list()
          mediaSaturated <- list()
          adstock <- check_adstock(adstock)

          for (v in 1:length(all_media)) {
            ################################################
            ## 1. Adstocking (whole data)
            m <- dt_modAdstocked[, all_media[v]][[1]]
            if (adstock == "geometric") {
              theta <- hypParamSam[paste0(all_media[v], "_thetas")][[1]][[1]]
              x_list <- adstock_geometric(x = m, theta = theta)
            } else if (adstock == "weibull_cdf") {
              shape <- hypParamSam[paste0(all_media[v], "_shapes")][[1]][[1]]
              scale <- hypParamSam[paste0(all_media[v], "_scales")][[1]][[1]]
              x_list <- adstock_weibull(x = m, shape = shape, scale = scale, type = "cdf")
            } else if (adstock == "weibull_pdf") {
              shape <- hypParamSam[paste0(all_media[v], "_shapes")][[1]][[1]]
              scale <- hypParamSam[paste0(all_media[v], "_scales")][[1]][[1]]
              x_list <- adstock_weibull(x = m, shape = shape, scale = scale, type = "pdf")
            }
            m_adstocked <- x_list$x_decayed
            mediaAdstocked[[v]] <- m_adstocked
            mediaVecCum[[v]] <- x_list$thetaVecCum

            # data.frame(id = rep(1:length(m), 2)) %>%
            #   mutate(value = c(m, m_adstocked),
            #          type = c(rep("raw", length(m)), rep("adstocked", length(m)))) %>%
            #   filter(id < 100) %>%
            #   ggplot(aes(x = id, y = value, colour = type)) +
            #   geom_line()

            ################################################
            ## 2. Saturation (only window data)
            m_adstockedRollWind <- m_adstocked[rollingWindowStartWhich:rollingWindowEndWhich]
            alpha <- hypParamSam[paste0(all_media[v], "_alphas")][[1]][[1]]
            gamma <- hypParamSam[paste0(all_media[v], "_gammas")][[1]][[1]]
            mediaSaturated[[v]] <- saturation_hill(m_adstockedRollWind, alpha = alpha, gamma = gamma)
            # plot(m_adstockedRollWind, mediaSaturated[[1]])
          }
          names(mediaAdstocked) <- names(mediaVecCum) <- names(mediaSaturated) <- all_media
          dt_modAdstocked <- dt_modAdstocked %>%
            select(-all_of(all_media)) %>%
            bind_cols(mediaAdstocked)
          mediaVecCum <- bind_cols(mediaVecCum)
          dt_modSaturated <- dt_modAdstocked[rollingWindowStartWhich:rollingWindowEndWhich, ] %>%
            select(-all_of(all_media)) %>%
            bind_cols(mediaSaturated)

          #####################################
          #### Split and prepare data for modelling

          dt_train <- dt_modSaturated

          ## Contrast matrix because glmnet does not treat categorical variables (one hot encoding)
          y_train <- dt_train$dep_var
          if (length(which(grepl("^[0-9]", dt_train))) > 1) {
            x_train <- model.matrix(dep_var ~ ., dt_train)[, -1]
          } else {
            x_train <- as.matrix(dt_train[, -1])
          }

          ## Define and set sign control
          dt_sign <- select(dt_modSaturated, -.data$dep_var)
          x_sign <- c(prophet_signs, context_signs, paid_media_signs, organic_signs)
          names(x_sign) <- c(prophet_vars, context_vars, paid_media_spends, organic_vars)
          check_factor <- sapply(dt_sign, is.factor)
          lower.limits <- upper.limits <- c()
          for (s in 1:length(check_factor)) {
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
            penalty.factor <- unlist(hypParamSamNG[i, grepl("penalty_", names(hypParamSamNG))])
          } else {
            penalty.factor <- rep(1, ncol(x_train))
          }

          glm_mod <- glmnet(
            x_train,
            y_train,
            family = "gaussian",
            alpha = 0, # 0 for ridge regression
            lambda = lambda_scaled,
            lower.limits = lower.limits,
            upper.limits = upper.limits,
            type.measure = "mse",
            penalty.factor = penalty.factor
          ) # plot(glm_mod); coef(glm_mod)

          # # When we used CV instead of nevergrad
          # lambda_range <- c(cvmod$lambda.min, cvmod$lambda.1se)
          # lambda <- lambda_range[1] + (lambda_range[2]-lambda_range[1]) * lambda_control

          #####################################
          #### Refit ridge regression with selected lambda from x-validation (intercept)

          ## If no lift calibration, refit using best lambda
          mod_out <- model_refit(x_train, y_train,
            lambda = lambda_scaled,
            lower.limits, upper.limits, intercept_sign
          )
          decompCollect <- model_decomp(
            coefs = mod_out$coefs,
            dt_modSaturated = dt_modSaturated,
            x = x_train,
            y_pred = mod_out$y_pred,
            i = i,
            dt_modRollWind = dt_modRollWind,
            refreshAddedStart = refreshAddedStart
          )
          nrmse <- mod_out$nrmse_train
          mape <- 0
          df.int <- mod_out$df.int

          #####################################
          #### get calibration mape

          if (!is.null(calibration_input)) {
            liftCollect <- calibrate_mmm(
              calibration_input, decompCollect,
              dayInterval = InputCollect$dayInterval
            )
            mape <- mean(liftCollect$mape_lift, na.rm = TRUE)
          }

          #####################################
          #### Calculate multi-objectives for pareto optimality

          ## DECOMP objective: sum of squared distance between decomp share and spend share to be minimized
          dt_decompSpendDist <- decompCollect$xDecompAgg %>%
            filter(.data$rn %in% paid_media_spends) %>%
            select(
              .data$rn, .data$xDecompAgg, .data$xDecompPerc, .data$xDecompMeanNon0Perc,
              .data$xDecompMeanNon0, .data$xDecompPercRF, .data$xDecompMeanNon0PercRF,
              .data$xDecompMeanNon0RF
            ) %>%
            left_join(select(
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

          # Calculate DECOMP.RSSD error
          if (!refresh) {
            decomp.rssd <- sqrt(sum((dt_decompSpendDist$effect_share - dt_decompSpendDist$spend_share)^2))
          } else {
            # xDecompAggPrev is NULL?
            dt_decompRF <- select(decompCollect$xDecompAgg, .data$rn, decomp_perc = .data$xDecompPerc) %>%
              left_join(select(xDecompAggPrev, .data$rn, decomp_perc_prev = .data$xDecompPerc),
                        by = "rn")
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

          ## adstock objective: sum of squared infinite sum of decay to be minimised - deprecated
          # dt_decaySum <- dt_mediaVecCum[,  .(rn = all_media, decaySum = sapply(.SD, sum)), .SDcols = all_media]
          # adstock.ssisd <- dt_decaySum[, sum(decaySum^2)]

          ## calibration objective: not calibration: mse, decomp.rssd, if calibration: mse, decom.rssd, mape_lift

          #####################################
          #### Collect output

          resultCollect <- list()
          common <- c(
            rsq_train = mod_out$rsq_train,
            nrmse = nrmse,
            decomp.rssd = decomp.rssd,
            mape = mape,
            lambda = lambda_scaled,
            lambda_hp = lambda_hp,
            lambda_max = lambda_max,
            lambda_min_ratio = lambda_min_ratio,
            iterPar = i,
            iterNG = lng,
            df.int = df.int
          )

          resultCollect[["resultHypParam"]] <- data.frame(hypParamSam) %>%
            select(-.data$lambda) %>%
            bind_cols(data.frame(t(common[1:8]))) %>%
            mutate(
              pos = prod(decompCollect$xDecompAgg$pos),
              Elapsed = as.numeric(difftime(Sys.time(), t1, units = "secs")),
              ElapsedAccum = as.numeric(difftime(Sys.time(), t0, units = "secs"))
            ) %>%
            bind_cols(data.frame(t(common[9:11]))) %>%
            dplyr::mutate_all(unlist)

          if (hyper_fixed) {
            resultCollect[["xDecompVec"]] <- decompCollect$xDecompVec %>%
              bind_cols(data.frame(t(common[1:8]))) %>%
              mutate(intercept = decompCollect$xDecompAgg$xDecompAgg[
                decompCollect$xDecompAgg$rn == "(Intercept)"]) %>%
              bind_cols(data.frame(t(common[9:11])))
          }

          resultCollect[["xDecompAgg"]] <- decompCollect$xDecompAgg %>%
            bind_cols(data.frame(t(common)))

          if (!is.null(calibration_input)) {
            resultCollect[["liftCalibration"]] <- liftCollect %>%
              bind_cols(data.frame(t(common)))
          }

          resultCollect[["decompSpendDist"]] <- dt_decompSpendDist %>%
            bind_cols(data.frame(t(common)))

          resultCollect <- append(resultCollect, as.list(common))

          if (cnt == iterTotal) {
            print(" === ")
            print(paste0(
              "Optimizer_name: ", optimizer_name, ";  Total_iterations: ",
              cnt, ";   Best MAPE: ", min(best_mape, mape)
            ))
          }
          return(resultCollect)
        }
      ) # end foreach parallel

      nrmse.collect <- sapply(doparCollect, function(x) x$nrmse)
      decomp.rssd.collect <- sapply(doparCollect, function(x) x$decomp.rssd)
      mape.lift.collect <- sapply(doparCollect, function(x) x$mape)

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

  resultCollect[["resultHypParam"]] <- bind_rows(
    lapply(resultCollectNG, function(x) {
      bind_rows(lapply(x, function(y) y$resultHypParam))
    })
  ) %>%
    arrange(.data$nrmse) %>%
    as_tibble()

  if (hyper_fixed) {
    resultCollect[["xDecompVec"]] <- bind_rows(
      lapply(resultCollectNG, function(x) {
        bind_rows(lapply(x, function(y) y$xDecompVec))
      })
    ) %>%
      arrange(.data$nrmse, .data$ds) %>%
      as_tibble()
  }

  resultCollect[["xDecompAgg"]] <- bind_rows(
    lapply(resultCollectNG, function(x) {
      bind_rows(lapply(x, function(y) y$xDecompAgg))
    })
  ) %>%
    arrange(.data$nrmse) %>%
    as_tibble()

  if (!is.null(calibration_input)) {
    resultCollect[["liftCalibration"]] <- bind_rows(
      lapply(resultCollectNG, function(x) {
        bind_rows(lapply(x, function(y) y$liftCalibration))
      })
    ) %>%
      arrange(.data$mape, .data$liftMedia, .data$liftStart) %>%
      as_tibble()
  }

  resultCollect[["decompSpendDist"]] <- bind_rows(
    lapply(resultCollectNG, function(x) {
      bind_rows(lapply(x, function(y) y$decompSpendDist))
    })
  ) %>%
    arrange(.data$nrmse) %>%
    as_tibble()

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

####################################################################
#' Response Function
#'
#' \code{robyn_response()} returns the response for a given
#' spend level of a given \code{paid_media_vars} from a selected model
#' result and selected model build (initial model, refresh model, etc.).
#'
#' @inheritParams robyn_allocator
#' @param media_metric A character. Selected media variable for the response.
#' Must be one value from paid_media_spends, paid_media_vars or organic_vars
#' @param metric_value Numeric. Desired metric value to return a response for.
#' @param dt_hyppar A data.frame. When \code{robyn_object} is not provided, use
#' \code{dt_hyppar = OutputCollect$resultHypParam}. It must be provided along
#' \code{select_model}, \code{dt_coef} and \code{InputCollect}.
#' @param dt_coef A data.frame. When \code{robyn_object} is not provided, use
#' \code{dt_coef = OutputCollect$xDecompAgg}. It must be provided along
#' \code{select_model}, \code{dt_hyppar} and \code{InputCollect}.
#' @examples
#' \dontrun{
#' # Having InputCollect and OutputCollect results OR robyn_object
#' # Set your exported model location
#' robyn_object <- "~/Desktop/MyRobyn.RDS"
#'
#' # Get marginal response (mResponse) and marginal ROI (mROI) for
#' # the next 1k on 80k for search_S, when provided the saved
#' # robyn_object by the robyn_save() function.
#' spend1 <- 80000
#' Response1 <- robyn_response(
#'   robyn_object = robyn_object,
#'   media_metric = "search_S",
#'   metric_value = spend1
#' )$response
#' # Get ROI for 80k
#' Response1 / spend1 # ROI for search 80k
#' # Get response for 81k
#' spend2 <- spend1 + 1000
#' Response2 <- robyn_response(
#'   robyn_object = robyn_object,
#'   media_metric = "search_S",
#'   metric_value = spend2
#' )$response
#' # Get ROI for 81k
#' Response2 / spend2 # ROI for search 81k
#' # Get marginal response (mResponse) for the next 1k on 80k
#' Response2 - Response1
#' # Get marginal ROI (mROI) for the next 1k on 80k
#' (Response2 - Response1) / (spend2 - spend1)
#'
#' # Example of getting paid media exposure response curves
#' imps <- 1000000
#' response_imps <- robyn_response(
#'   robyn_object = robyn_object,
#'   media_metric = "facebook_I",
#'   metric_value = imps
#' )$response
#' response_per_1k_imps <- response_imps / imps * 1000
#' response_per_1k_imps
#'
#' # Example of getting organic media exposure response curves
#' sendings <- 30000
#' response_sending <- robyn_response(
#'   robyn_object = robyn_object,
#'   media_metric = "newsletter",
#'   metric_value = sendings
#' )$response
#' response_per_1k_send <- response_sending / sendings * 1000
#' response_per_1k_send
#'
#' # Get response for 80k for search_S from the third model refresh
#' robyn_response(
#'   robyn_object = robyn_object,
#'   select_build = 3,
#'   media_metric = "search_S",
#'   metric_value = 80000
#' )
#'
#' # Get response for 80k for search_S from the a certain model SolID
#' # in the current model output in the global environment
#' robyn_response(
#'   media_metric = "search_S",
#'   metric_value = 80000,
#'   select_model = "3_10_3",
#'   dt_hyppar = OutputCollect$resultHypParam,
#'   dt_coef = OutputCollect$xDecompAgg,
#'   InputCollect = InputCollect
#' )
#' }
#' @return List. Response value and plot. Class: \code{robyn_response}.
#' @export
robyn_response <- function(robyn_object = NULL,
                           select_build = NULL,
                           media_metric = NULL,
                           select_model = NULL,
                           metric_value = NULL,
                           dt_hyppar = NULL,
                           dt_coef = NULL,
                           InputCollect = NULL,
                           OutputCollect = NULL,
                           quiet = FALSE) {

  ## Get input
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
      if (!quiet & length(select_build_all) > 1) {
        message(
          "Using latest model: ", ifelse(select_build == 0, "initial model", paste0("refresh model #", select_build)),
          " for the response function. Use parameter 'select_build' to specify which run to use"
        )
      }
    }

    if (!(select_build %in% select_build_all) | length(select_build) != 1) {
      stop("'select_build' must be one value of ", paste(select_build_all, collapse = ", "))
    }

    listName <- ifelse(select_build == 0, "listInit", paste0("listRefresh", select_build))
    InputCollect <- Robyn[[listName]][["InputCollect"]]
    OutputCollect <- Robyn[[listName]][["OutputCollect"]]
    dt_hyppar <- OutputCollect$resultHypParam
    dt_coef <- OutputCollect$xDecompAgg
    select_model <- OutputCollect$selectID
  } else {
    # Try to get some pre-filled values
    if (is.null(dt_hyppar)) dt_hyppar <- OutputCollect$resultHypParam
    if (is.null(dt_coef)) dt_coef <- OutputCollect$xDecompAgg
    if (any(is.null(dt_hyppar), is.null(dt_coef), is.null(InputCollect), is.null(OutputCollect))) {
      stop("When 'robyn_object' is not provided, 'InputCollect' & 'OutputCollect' must be provided")
    }
  }

  ## Prep environment
  if (TRUE) {
    dt_input <- InputCollect$dt_input
    startRW <- InputCollect$rollingWindowStartWhich
    endRW <- InputCollect$rollingWindowEndWhich
    adstock <- InputCollect$adstock
    spendExpoMod <- InputCollect$modNLSCollect
    paid_media_vars <- InputCollect$paid_media_vars
    paid_media_spends <- InputCollect$paid_media_spends
    exposure_vars <- InputCollect$exposure_vars
    organic_vars <- InputCollect$organic_vars
    allSolutions <- unique(dt_hyppar$solID)
  }

  if (!(select_model %in% allSolutions)) {
    stop(paste0(
      "Input 'select_model' must be one of these values: ",
      paste(allSolutions, collapse = ", ")
    ))
  }

  ## Get media values
  if (media_metric %in% paid_media_spends & length(media_metric) == 1) {
    metric_type <- "spend"
  } else if (media_metric %in% exposure_vars & length(media_metric) == 1) {
    metric_type <- "exposure"
  } else if (media_metric %in% organic_vars & length(media_metric) == 1) {
    metric_type <- "organic"
  } else {
    stop(paste(
      "Invalid 'media_metric' input. It must be any media variable from",
      "paid_media_spends (spend), paid_media_vars (exposure),",
      "or organic_vars (organic); NOT:", media_metric,
      paste("\n- paid_media_spends:", v2t(paid_media_spends, quotes = FALSE)),
      paste("\n- paid_media_vars:", v2t(paid_media_vars, quotes = FALSE)),
      paste("\n- organic_vars:", v2t(organic_vars, quotes = FALSE))
    ))
  }

  if (is.nan(metric_value)) metric_value <- NULL
  check_metric_value(metric_value, media_metric)

  ## Transform exposure to spend when necessary
  if (metric_type == "exposure") {
    get_spend_name <- paid_media_spends[which(paid_media_vars == media_metric)]
    expo_vec <- dt_input[, media_metric][[1]]
    # Use non-0 mean as marginal level if metric_value not provided
    if (is.null(metric_value)) {
      metric_value <- mean(expo_vec[startRW:endRW][expo_vec[startRW:endRW] > 0])
      if (!quiet) message("Input 'metric_value' not provided. Using mean of ", media_metric, " instead")
    }
    # Fit spend to exposure
    spend_vec <- dt_input[, get_spend_name][[1]]
    nls_select <- filter(spendExpoMod, .data$channel == media_metric, .data$rsq_nls > .data$rsq_lm)
    if (nls_select) {
      Vmax <- spendExpoMod$Vmax[spendExpoMod$channel == media_metric]
      Km <- spendExpoMod$Km[spendExpoMod$channel == media_metric]
      media_vec <- mic_men(x = spend_vec, Vmax = Vmax, Km = Km, reverse = FALSE)
    } else {
      coef_lm <- spendExpoMod$coef_lm[spendExpoMod$channel == media_metric]
      media_vec <- spend_vec * coef_lm
    }
    hpm_name <- get_spend_name
  } else {
    media_vec <- dt_input[, media_metric][[1]]
    # use non-0 means marginal level if spend not provided
    if (is.null(metric_value)) {
      metric_value <- mean(media_vec[startRW:endRW][media_vec[startRW:endRW] > 0])
      if (!quiet) message("Input 'metric_value' not provided. Using mean of ", media_metric, " instead")
    }
    hpm_name <- media_metric
  }

  ## Adstocking
  if (adstock == "geometric") {
    theta <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_thetas")]]
    x_list <- adstock_geometric(x = media_vec, theta = theta)
  } else if (adstock == "weibull_cdf") {
    shape <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_shapes")]]
    scale <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_scales")]]
    x_list <- adstock_weibull(x = media_vec, shape = shape, scale = scale, type = "cdf")
  } else if (adstock == "weibull_pdf") {
    shape <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_shapes")]]
    scale <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_scales")]]
    x_list <- adstock_weibull(x = media_vec, shape = shape, scale = scale, type = "pdf")
  }
  m_adstocked <- x_list$x_decayed

  ## Saturation
  m_adstockedRW <- m_adstocked[startRW:endRW]
  alpha <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_alphas")]]
  gamma <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_gammas")]]
  Saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma, x_marginal = metric_value)
  m_saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma)

  ## Decomp
  coeff <- dt_coef[dt_coef$solID == select_model & dt_coef$rn == hpm_name, ][["coef"]]
  response_vec <- m_saturated * coeff
  Response <- as.numeric(Saturated * coeff)

  ## Plot optimal response
  media_type <- ifelse(metric_type == "organic", "organic", "paid")
  dt_line <- data.frame(metric = m_adstockedRW, response = response_vec, channel = media_metric)
  dt_point <- data.frame(input = metric_value, output = Response)
  p_res <- ggplot(dt_line, aes(x = .data$metric, y = .data$response)) +
    geom_line(color = "steelblue") +
    geom_point(data = dt_point, aes(x = .data$input, y = .data$output), size = 3) +
    labs(
      title = paste(
        "Saturation curve of", media_type, "media:", media_metric,
        ifelse(metric_type == "spend", "spend metric", "exposure metric")
      ),
      subtitle = sprintf(
        "Response of %s @ %s",
        formatNum(dt_point$output, signif = 4),
        formatNum(dt_point$input, signif = 4)
      ),
      x = "Metric", y = "Response"
    ) +
    theme_lares() +
    scale_x_abbr() +
    scale_y_abbr()

  class(Response) <- unique(c("robyn_response", class(Response)))
  return(list(
    response = Response,
    plot = p_res
  ))
}

model_decomp <- function(coefs, dt_modSaturated, x, y_pred, i, dt_modRollWind, refreshAddedStart) {

  ## Input for decomp
  y <- dt_modSaturated$dep_var
  x <- data.frame(x)
  indepVar <- select(dt_modSaturated, -.data$dep_var)
  intercept <- coefs[1]
  indepVarName <- names(indepVar)
  indepVarCat <- indepVarName[sapply(indepVar, is.factor)]

  ## Decomp x
  xDecomp <- data.frame(mapply(function(regressor, coeff) {
    regressor * coeff
  }, regressor = x, coeff = coefs[-1]))
  xDecomp <- cbind(data.frame(intercept = rep(intercept, nrow(xDecomp))), xDecomp)
  xDecompOut <- cbind(data.frame(ds = dt_modRollWind$ds, y = y, y_pred = y_pred), xDecomp)

  ## QA decomp
  y_hat <- rowSums(xDecomp, na.rm = TRUE)
  errorTerm <- y_hat - y_pred
  if (prod(round(y_pred) == round(y_hat)) == 0) {
    message(paste0(
      "Attention for loop ", i,
      ": manual decomp is not matching linear model prediction. ",
      "Deviation is ", round(mean(errorTerm / y) * 100, 2), "%"
    ))
  }

  ## Output decomp
  y_hat.scaled <- rowSums(abs(xDecomp), na.rm = TRUE)
  xDecompOutPerc.scaled <- abs(xDecomp) / y_hat.scaled
  xDecompOut.scaled <- y_hat * xDecompOutPerc.scaled

  temp <- select(xDecompOut, .data$intercept, all_of(indepVarName))
  xDecompOutAgg <- sapply(temp, function(x) sum(x))
  xDecompOutAggPerc <- xDecompOutAgg / sum(y_hat)
  xDecompOutAggMeanNon0 <- sapply(temp, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x != 0])))
  xDecompOutAggMeanNon0[is.nan(xDecompOutAggMeanNon0)] <- 0
  xDecompOutAggMeanNon0Perc <- xDecompOutAggMeanNon0 / sum(xDecompOutAggMeanNon0)

  refreshAddedStartWhich <- which(xDecompOut$ds == refreshAddedStart)
  refreshAddedEnd <- max(xDecompOut$ds)
  refreshAddedEndWhich <- which(xDecompOut$ds == refreshAddedEnd)

  temp <- select(xDecompOut, .data$intercept, all_of(indepVarName)) %>%
    slice(refreshAddedStartWhich:refreshAddedEndWhich)
  xDecompOutAggRF <- sapply(temp, function(x) sum(x))
  y_hatRF <- y_hat[refreshAddedStartWhich:refreshAddedEndWhich]
  xDecompOutAggPercRF <- xDecompOutAggRF / sum(y_hatRF)
  xDecompOutAggMeanNon0RF <- sapply(temp, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x != 0])))
  xDecompOutAggMeanNon0RF[is.nan(xDecompOutAggMeanNon0RF)] <- 0
  xDecompOutAggMeanNon0PercRF <- xDecompOutAggMeanNon0RF / sum(xDecompOutAggMeanNon0RF)

  coefsOutCat <- coefsOut <- data.frame(rn = rownames(coefs), coefs)
  if (length(indepVarCat) > 0) {
    coefsOut$rn <- sapply(indepVarCat, function(x) str_replace(coefsOut$rn, paste0(x, ".*"), x))
  }
  coefsOut <- coefsOut %>%
    group_by(.data$rn) %>%
    summarise(s0 = mean(.data$s0)) %>%
    rename("coef" = "s0") %>%
    .[match(rownames(coefsOut), .$rn), ]

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
    xDecompAgg = decompOutAgg, coefsOutCat = coefsOutCat
  )

  return(decompCollect)
}


calibrate_mmm <- function(calibration_input, decompCollect, dayInterval) {

  ## Prep lift inputs
  getDecompVec <- decompCollect$xDecompVec
  getLiftMedia <- unique(calibration_input$channel)
  liftCollect <- list()

  # Loop per lift channel
  for (m in seq_along(getLiftMedia)) {
    liftWhich <- str_which(calibration_input$channel, getLiftMedia[m])
    liftCollect2 <- list()

    # Loop per lift test per channel
    for (lw in seq_along(liftWhich)) {

      ## Get lift period subset
      liftStart <- calibration_input$liftStartDate[liftWhich[lw]]
      liftEnd <- calibration_input$liftEndDate[liftWhich[lw]]
      df <- filter(getDecompVec, .data$ds >= liftStart, .data$ds <= liftEnd)
      liftPeriodVec <- select(df, .data$ds, getLiftMedia[m])
      liftPeriodVecDependent <- select(df, .data$ds, .data$y)

      ## Scale decomp
      mmmDays <- nrow(liftPeriodVec) * dayInterval
      liftDays <- as.integer(liftEnd - liftStart + 1)
      y_hatLift <- sum(unlist(getDecompVec[, -1])) # Total pred sales
      x_decompLift <- sum(liftPeriodVec[, 2])
      x_decompLiftScaled <- x_decompLift / mmmDays * liftDays
      y_scaledLift <- sum(liftPeriodVecDependent$y) / mmmDays * liftDays

      ## Output
      liftCollect2[[lw]] <- data.frame(
        liftMedia = getLiftMedia[m],
        liftStart = liftStart,
        liftEnd = liftEnd,
        liftAbs = calibration_input$liftAbs[liftWhich[lw]],
        decompAbsScaled = x_decompLiftScaled,
        dependent = y_scaledLift
      )
    }
    liftCollect[[m]] <- bind_rows(liftCollect2)
  }

  ## Get mape_lift -> Then MAPE = mean(mape_lift)
  liftCollect <- bind_rows(liftCollect) %>%
    mutate(mape_lift = abs((.data$decompAbsScaled - .data$liftAbs) / .data$liftAbs))
  return(liftCollect)
}


model_refit <- function(x_train, y_train, lambda, lower.limits, upper.limits, intercept_sign = "non_negative") {
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

  df.int <- 1

  ## drop intercept if negative and intercept_sign == "non_negative"
  if (intercept_sign == "non_negative" & coef(mod)[1] < 0) {
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
    df.int <- 0
  } # ; plot(mod); print(mod)

  y_trainPred <- predict(mod, s = lambda, newx = x_train)
  rsq_train <- get_rsq(true = y_train, predicted = y_trainPred, p = ncol(x_train), df.int = df.int)

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
    rsq_train = rsq_train,
    nrmse_train = nrmse_train,
    coefs = coefs,
    y_pred = as.vector(y_trainPred),
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

hyper_collector <- function(InputCollect, hyper_in, add_penalty_factor, dt_hyper_fixed = NULL, cores) {

  # Fetch hyper-parameters based on media
  hypParamSamName <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)

  # Add lambda
  hypParamSamName <- c(hypParamSamName, "lambda")

  # Add penalty factor hyper-parameters names
  for_penalty <- names(select(InputCollect$dt_mod, -.data$ds, -.data$dep_var))
  if (add_penalty_factor) hypParamSamName <- c(hypParamSamName, paste0("penalty_", for_penalty))

  # Check hyper_fixed condition
  all_fixed <- check_hyper_fixed(InputCollect, dt_hyper_fixed, add_penalty_factor)

  if (!all_fixed) {

    # Collect media hyperparameters
    hyper_bound_list <- list()
    for (i in 1:length(hypParamSamName)) {
      hyper_bound_list[i] <- hyper_in[hypParamSamName[i]]
      names(hyper_bound_list)[i] <- hypParamSamName[i]
    }

    # Add unfixed lambda hyperparameters manually
    if (length(hyper_bound_list[["lambda"]]) != 1) {
      hyper_bound_list$lambda <- c(0, 1)
    }

    # Add unfixed penalty.factor hyperparameters manually
    penalty_names <- paste0("penalty_", for_penalty)
    if (add_penalty_factor) {
      for (penalty in penalty_names) {
        if (length(hyper_bound_list[[penalty]]) != 1) {
          hyper_bound_list[[penalty]] <- c(0, 1)
        }
      }
    }

    # Get hyperparameters for Nevergrad
    hyper_bound_list_updated <- hyper_bound_list[which(sapply(hyper_bound_list, length) == 2)]

    # Get fixed hyperparameters
    hyper_bound_list_fixed <- hyper_bound_list[which(sapply(hyper_bound_list, length) == 1)]

    hyper_list_bind <- c(hyper_bound_list_updated, hyper_bound_list_fixed)
    hyper_list_all <- list()
    for (i in 1:length(hypParamSamName)) {
      hyper_list_all[[i]] <- hyper_list_bind[[hypParamSamName[i]]]
      names(hyper_list_all)[i] <- hypParamSamName[i]
    }

    dt_hyper_fixed_mod <- data.frame(sapply(hyper_bound_list_fixed, function(x) rep(x, cores)))
  } else {
    hyper_bound_list_fixed <- list()
    for (i in 1:length(hypParamSamName)) {
      hyper_bound_list_fixed[[i]] <- dt_hyper_fixed[[hypParamSamName[i]]]
      names(hyper_bound_list_fixed)[i] <- hypParamSamName[i]
    }

    hyper_list_all <- hyper_bound_list_fixed
    hyper_bound_list_updated <- hyper_bound_list_fixed[which(sapply(hyper_bound_list_fixed, length) == 2)]
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

init_msgs_run <- function(InputCollect, refresh, lambda_control, quiet = FALSE) {
  if (!is.null(lambda_control)) {
    message("'lambda_control' deprecated in v3.6.0; lambda is now selected by hyperparameter optimisation")
  }
  if (!quiet) {
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
}
