robyn_mmm <- function(InputCollect,
                      hyper_collect,
                      iterations,
                      cores,
                      nevergrad_algo,
                      intercept = TRUE,
                      intercept_sign,
                      ts_validation = TRUE,
                      add_penalty_factor = FALSE,
                      objective_weights = NULL,
                      dt_hyper_fixed = NULL,
                      # lambda_fixed = NULL,
                      rssd_zero_penalty = TRUE,
                      refresh = FALSE,
                      trial = 1L,
                      seed = 123L,
                      quiet = FALSE, ...) {
  if (iterations > 1) {
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
    paid_media_selected <- InputCollect$paid_media_selected
    exposure_vars <- InputCollect$exposure_vars
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
    rn = paid_media_selected,
    total_spend = unlist(summarise_all(temp, sum)),
    # mean_spend = unlist(summarise_all(temp, function(x) {
    #   ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0]))
    # }))
    mean_spend = unlist(summarise_all(temp, mean))
  ) %>%
    mutate(spend_share = .data$total_spend / sum(.data$total_spend))
  if (length(c(exposure_vars, organic_vars)) > 0) {
    temp <- select(dt_inputTrain, all_of(c(exposure_vars, organic_vars))) %>% summarise_all(mean) %>% unlist
    temp <- data.frame(rn = c(exposure_vars, organic_vars), mean_exposure = temp)
    dt_spendShare <- full_join(dt_spendShare, temp, by = "rn")
  } else {
    dt_spendShare$mean_exposure <- NA
  }
  # When not refreshing, dt_spendShareRF = dt_spendShare
  refreshAddedStartWhich <- which(dt_modRollWind$ds == refreshAddedStart)
  temp <- select(dt_inputTrain, all_of(paid_media_spends)) %>%
    slice(refreshAddedStartWhich:rollingWindowLength)
  dt_spendShareRF <- data.frame(
    rn = paid_media_selected,
    total_spend = unlist(summarise_all(temp, sum)),
    # mean_spend = unlist(summarise_all(temp, function(x) {
    #   ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0]))
    # }))
    mean_spend = unlist(summarise_all(temp, mean))
  ) %>%
    mutate(spend_share = .data$total_spend / sum(.data$total_spend))
  # Join both dataframes into a single one
  if (length(c(exposure_vars, organic_vars)) > 0) {
    temp <- select(dt_inputTrain,  all_of(c(exposure_vars, organic_vars))) %>%
      slice(refreshAddedStartWhich:rollingWindowLength) %>%
      summarise_all(mean) %>% unlist
    temp <- data.frame(rn = c(exposure_vars, organic_vars), mean_exposure = temp)
    dt_spendShareRF <- full_join(dt_spendShareRF, temp, by = "rn")
  } else {
    dt_spendShareRF$mean_exposure <- NA
  }
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
      if (is.null(objective_weights)) {
        objective_weights <- tuple(1, 1)
      } else {
        objective_weights <- tuple(objective_weights[1], objective_weights[2])
      }
      optimizer$set_objective_weights(objective_weights)
    } else {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1, 1, 1))
      if (is.null(objective_weights)) {
        objective_weights <- tuple(1, 1, 1)
      } else {
        objective_weights <- tuple(objective_weights[1], objective_weights[2], objective_weights[3])
      }
      optimizer$set_objective_weights(objective_weights)
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
                hyppar_value <- signif(nevergrad_hp_val[[co]][index], 10)
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
            temp <- run_transformations(all_media = InputCollect$all_media,
                                        window_start_loc = InputCollect$rollingWindowStartWhich,
                                        window_end_loc = InputCollect$rollingWindowEndWhich,
                                        dt_mod = InputCollect$dt_mod,
                                        adstock = InputCollect$adstock,
                                        dt_hyppar = hypParamSam, ...)

            #####################################
            #### Split train & test and prepare data for modelling

            dt_window <- temp$dt_modSaturated

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
            names(x_sign) <- c(prophet_vars, context_vars, paid_media_selected, organic_vars)
            check_factor <- unlist(lapply(dt_sign, is.factor))
            lower.limits <- rep(0, length(prophet_signs))
            upper.limits <- rep(1, length(prophet_signs))
            trend_loc <- which(colnames(x_train) == "trend")
            if (length(trend_loc) > 0 & sum(x_train[, trend_loc]) < 0) {
              trend_loc <- which(prophet_vars == "trend")
              lower.limits[trend_loc] <- -1
              upper.limits[trend_loc] <- 0
            }
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
              x_train = x_train,
              y_train = y_train,
              x_val = x_val,
              y_val = y_val,
              x_test = x_test,
              y_test = y_test,
              lambda = lambda_scaled,
              lower.limits = lower.limits,
              upper.limits = upper.limits,
              intercept = intercept,
              intercept_sign = intercept_sign,
              penalty.factor = penalty.factor,
              ...
            )
            decompCollect <- model_decomp(
              inputs = list(
                coefs = mod_out$coefs,
                y_pred = mod_out$y_pred,
                dt_modSaturated = temp$dt_modSaturated,
                dt_saturatedImmediate = temp$dt_saturatedImmediate,
                dt_saturatedCarryover = temp$dt_saturatedCarryover,
                dt_modRollWind = dt_modRollWind,
                refreshAddedStart = refreshAddedStart
              ))
            nrmse <- ifelse(ts_validation, mod_out$nrmse_val, mod_out$nrmse_train)
            mape <- 0
            df.int <- mod_out$df.int

            #####################################
            #### MAPE: Calibration error
            if (!is.null(calibration_input)) {
              liftCollect <- lift_calibration(
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
            dt_loss_calc <- decompCollect$xDecompAgg %>%
              filter(.data$rn %in% c(paid_media_selected, organic_vars)) %>%
              select(
                .data$rn, .data$xDecompPerc, .data$xDecompPercRF
              ) %>% left_join(
              select(
                dt_spendShare,
                c("rn", "spend_share", "spend_share_refresh","mean_spend",
                  "total_spend", "mean_exposure", "mean_exposure_refresh")
              ),
              by = "rn"
            )
            dt_loss_calc <- bind_rows(
              dt_loss_calc %>% filter(.data$rn %in% paid_media_selected) %>%
                mutate(
                  effect_share = .data$xDecompPerc / sum(.data$xDecompPerc),
                  effect_share_refresh = .data$xDecompPercRF / sum(.data$xDecompPercRF)
                ),
              dt_loss_calc %>% filter(.data$rn %in% organic_vars) %>%
                mutate(
                  effect_share = NA, effect_share_refresh = NA)
            ) %>% select(-c("xDecompPerc", "xDecompPercRF"))
            decompCollect$xDecompAgg <- left_join(
              decompCollect$xDecompAgg, dt_loss_calc, by = "rn")
            dt_loss_calc <- dt_loss_calc %>% filter(.data$rn %in% paid_media_selected)
            if (!refresh) {
              decomp.rssd <- sqrt(sum((dt_loss_calc$effect_share - dt_loss_calc$spend_share)^2))
              # Penalty for models with more 0-coefficients
              if (rssd_zero_penalty) {
                is_0eff <- round(dt_loss_calc$effect_share, 4) == 0
                share_0eff <- sum(is_0eff) / length(dt_loss_calc$effect_share)
                decomp.rssd <- decomp.rssd * (1 + share_0eff)
              }
            } else {
              dt_decompRF <- select(decompCollect$xDecompAgg, .data$rn, decomp_perc = .data$xDecompPerc) %>%
                left_join(select(xDecompAggPrev, .data$rn, decomp_perc_prev = .data$xDecompPerc),
                  by = "rn"
                )
              decomp.rssd.media <- dt_decompRF %>%
                filter(.data$rn %in% paid_media_selected) %>%
                summarise(rssd.media = sqrt(mean((.data$decomp_perc - .data$decomp_perc_prev)^2))) %>%
                pull(.data$rssd.media)
              decomp.rssd.nonmedia <- dt_decompRF %>%
                filter(!.data$rn %in% paid_media_selected) %>%
                summarise(rssd.nonmedia = sqrt(mean((.data$decomp_perc - .data$decomp_perc_prev)^2))) %>%
                pull(.data$rssd.nonmedia)
              decomp.rssd <- decomp.rssd.media + decomp.rssd.nonmedia /
                (1 - refresh_steps / rollingWindowLength)
            }
            # When all media in this iteration have 0 coefficients
            if (is.nan(decomp.rssd)) {
              decomp.rssd <- Inf
              decompCollect$xDecompAgg <- decompCollect$xDecompAgg %>%
                mutate(effect_share = ifelse(is.na(.data$effect_share), NA, 0))
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
              bind_cols(as_tibble(t(temp$inflexions))) %>%
              bind_cols(as_tibble(t(temp$inflations))) %>%
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

            # resultCollect[["decompSpendDist"]] <- dt_decompSpendDist %>%
            #   bind_cols(common)
            resultCollect <- append(resultCollect, as.list(common))
            return(resultCollect)
          }

          ########### Parallel start
          nrmse.collect <- NULL
          decomp.rssd.collect <- NULL
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
              doparCollect <- foreach(i = 1:iterPar, .options.RNG = seed) %dorng% robyn_iterations(i)
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
  if (cores > 1) {
    stopImplicitCluster()
    registerDoSEQ()
    getDoParWorkers()
  }

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

  # resultCollect[["decompSpendDist"]] <- as_tibble(bind_rows(
  #   lapply(resultCollectNG, function(x) {
  #     bind_rows(lapply(x, function(y) y$decompSpendDist))
  #   })
  # ))

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