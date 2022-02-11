# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Includes function robyn_allocator()

####################################################################
#' Robyn budget allocator
#'
#' The \code{robyn_allocator()} function returns a new split of media
#' variable spends that maximizes the total media response.
#'
#' @inheritParams robyn_run
#' @inheritParams robyn_outputs
#' @param robyn_object Character. Path of the \code{Robyn.RDS} object
#' that contains all previous modeling information.
#' @param select_build Integer. Default to the latest model build. \code{select_build = 0}
#' selects the initial model. \code{select_build = 1} selects the first refresh model.
#' @param InputCollect List. Contains all input parameters for the model.
#' Required when \code{robyn_object} is not provided.
#' @param OutputCollect List. Containing all model result.
#' Required when \code{robyn_object} is not provided.
#' @param select_model Character. A model \code{SolID}. When \code{robyn_object}
#' is provided, \code{select_model} defaults to the already selected \code{SolID}. When
#' \code{robyn_object} is not provided, \code{select_model} must be provided with
#' \code{InputCollect} and \code{OutputCollect}, and must be one of
#' \code{OutputCollect$allSolutions}.
#' @param optim_algo Character. Default to \code{"SLSQP_AUGLAG"}, short for "Sequential Least-Squares
#' Quadratic Programming" and "Augmented Lagrangian". Alternatively, "\code{"MMA_AUGLAG"},
#' short for "Methods of Moving Asymptotes". More details see the documentation of
#' NLopt \href{https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/}{here}.
#' @param scenario Character. Accepted options are: \code{"max_historical_response"} or
#' \code{"max_response_expected_spend"}. \code{"max_historical_response"} simulates the scenario
#' "what's the optimal media spend allocation given the same average spend level in history?",
#' while \code{"max_response_expected_spend"} simulates the scenario "what's the optimal media
#' spend allocation of a given future spend level for a given period?"
#' @param expected_spend Numeric. The expected future spend volume. Only applies when
#' \code{scenario = "max_response_expected_spend"}.
#' @param expected_spend_days Integer. The duration of the future spend volume in
#' \code{expected_spend}. Only applies when \code{scenario = "max_response_expected_spend"}.
#' @param channel_constr_low,channel_constr_up Numeric vector. The lower and upper bounds
#' for each paid media variable when maximizing total media response. \code{channel_constr_low
#' = 0.7} means minimum spend of the variable is 70% of historical average. Lower bound must
#' be >=0.01. \code{channel_constr_up = 1.5} means maximum spend of the variable is 150% of
#' historical average. Upper bound must be >= lower bound. Both must have same length and order
#' as \code{paid_media_vars}. nIt's ot recommended to 'exaggerate' upper bounds, esp. if the
#' new level is way higher than historical level.
#' @param maxeval Integer. The maximum iteration of the global optimization algorithm.
#' Defaults to 100000.
#' @param constr_mode Character. Options are \code{"eq"} or \code{"ineq"},
#' indicating constraints with equality or inequality.
#' @return A list object containing allocator result.
#' @examples
#' \dontrun{
#' # Check media summary for selected model from the simulated data
#' select_model <- "3_10_3"
#' OutputCollect$xDecompAgg[
#'   solID == select_model & !is.na(mean_spend),
#'   .(rn, coef, mean_spend, mean_response, roi_mean,
#'     total_spend,
#'     total_response = xDecompAgg, roi_total, solID
#'   )
#' ]
#'
#' # Run allocator with 'InputCollect' and 'OutputCollect'
#' # with 'scenario = "max_historical_response"'
#' AllocatorCollect <- robyn_allocator(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   select_model = select_model,
#'   scenario = "max_historical_response",
#'   channel_constr_low = c(0.7, 0.7, 0.7, 0.7, 0.7),
#'   channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5)
#' )
#'
#' # Run allocator with a 'robyn_object' from the second model refresh
#' # with 'scenario = "max_response_expected_spend"'
#' AllocatorCollect <- robyn_allocator(
#'   robyn_object = robyn_object,
#'   select_build = 2,
#'   scenario = "max_response_expected_spend",
#'   channel_constr_low = c(0.7, 0.7, 0.7, 0.7, 0.7),
#'   channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5),
#'   expected_spend = 100000,
#'   expected_spend_days = 90
#' )
#' }
#' @export
robyn_allocator <- function(robyn_object = NULL,
                            select_build = NULL,
                            InputCollect = NULL,
                            OutputCollect = NULL,
                            select_model = NULL,
                            optim_algo = "SLSQP_AUGLAG",
                            scenario = "max_historical_response",
                            expected_spend = NULL,
                            expected_spend_days = NULL,
                            channel_constr_low = 0.5,
                            channel_constr_up = 2,
                            maxeval = 100000,
                            constr_mode = "eq",
                            export = TRUE,
                            quiet = FALSE,
                            ui = FALSE) {

  #####################################
  #### Set local environment

  ## Collect input
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
        "Using latest model: ", ifelse(select_build == 0, "initial model", paste0("refresh model #", select_build
        )),
        " for the response function. select_build = 0 selects initial model, 1 the first refresh etc"
      )
    }
    if (!(select_build %in% select_build_all) | length(select_build) != 1) {
      stop("Input 'select_build' must be one value of ", paste(select_build_all, collapse = ", "))
    }
    listName <- ifelse(select_build == 0, "listInit", paste0("listRefresh", select_build))
    InputCollect <- Robyn[[listName]][["InputCollect"]]
    OutputCollect <- Robyn[[listName]][["OutputCollect"]]
    select_model <- OutputCollect$selectID
  } else if (any(is.null(InputCollect), is.null(OutputCollect), is.null(select_model))) {
    stop("When 'robyn_object' is not provided, then InputCollect, OutputCollect, select_model must be provided")
  }

  message(paste(">>> Running budget allocator for model ID", select_model, "..."))

  ## Set local data & params values
  if (TRUE) {
    dt_input <- InputCollect$dt_input
    dt_mod <- InputCollect$dt_mod
    paid_media_vars <- InputCollect$paid_media_vars
    media_order <- order(paid_media_vars)
    paid_media_spends <- InputCollect$paid_media_spends
    mediaVarSorted <- paid_media_vars[media_order]
    mediaSpendSorted <- paid_media_spends[media_order]
    exposureVarName <- InputCollect$exposureVarName
    startRW <- InputCollect$rollingWindowStartWhich
    endRW <- InputCollect$rollingWindowEndWhich
    adstock <- InputCollect$adstock
    spendExpoMod <- InputCollect$modNLSCollect
  }

  ## Check inputs and parameters
  check_allocator(OutputCollect, select_model, paid_media_vars, scenario,
                  channel_constr_low, channel_constr_up,
                  expected_spend, expected_spend_days, constr_mode)
  names(channel_constr_low) <- paid_media_vars
  names(channel_constr_up) <- paid_media_vars
  dt_hyppar <- OutputCollect$resultHypParam[solID == select_model]
  dt_bestCoef <- OutputCollect$xDecompAgg[solID == select_model & rn %in% paid_media_vars]
  dt_mediaSpend <- dt_input[startRW:endRW, mediaSpendSorted, with = FALSE]

  ## Sort table and get filter for channels mmm coef reduced to 0
  dt_coef <- dt_bestCoef[, .(rn, coef)]
  get_rn_order <- order(dt_bestCoef$rn)
  dt_coefSorted <- dt_coef[get_rn_order]
  dt_bestCoef <- dt_bestCoef[get_rn_order]
  coefSelectorSorted <- dt_coefSorted[, coef > 0]
  names(coefSelectorSorted) <- dt_coefSorted$rn

  ## Filter and sort all variables by name that is essential for the apply function later
  mediaVarSortedFiltered <- mediaVarSorted[coefSelectorSorted]
  mediaSpendSortedFiltered <- mediaSpendSorted[coefSelectorSorted]
  if (!all(coefSelectorSorted)) {
    chn_coef0 <- setdiff(mediaVarSorted, mediaVarSortedFiltered)
    message(paste(chn_coef0, collapse = ", "), " are excluded in optimiser because their coeffients are 0")
  }
  dt_hyppar <- dt_hyppar[, .SD, .SDcols = na.omit(
    str_extract(names(dt_hyppar), paste(paste0(mediaVarSortedFiltered, ".*"), collapse = "|"))
  )]
  setcolorder(dt_hyppar, sort(names(dt_hyppar)))
  dt_optim <- dt_mod[, mediaVarSortedFiltered, with = FALSE]
  dt_optimCost <- dt_input[startRW:endRW, mediaSpendSortedFiltered, with = FALSE]
  dt_bestCoef <- dt_bestCoef[rn %in% mediaVarSortedFiltered]
  costMultiplierVec <- InputCollect$mediaCostFactor[mediaVarSortedFiltered]

  if (any(InputCollect$costSelector)) {
    dt_modNLS <- merge(data.table(channel = mediaVarSortedFiltered), spendExpoMod, all.x = TRUE, by = "channel")
    vmaxVec <- dt_modNLS[order(rank(channel))][, Vmax]
    names(vmaxVec) <- mediaVarSortedFiltered
    kmVec <- dt_modNLS[order(rank(channel))][, Km]
    names(kmVec) <- mediaVarSortedFiltered
  } else {
    vmaxVec <- rep(0, length(mediaVarSortedFiltered))
    kmVec <- rep(0, length(mediaVarSortedFiltered))
  }

  costSelectorSorted <- InputCollect$costSelector[media_order]
  costSelectorSorted <- costSelectorSorted[coefSelectorSorted]
  costSelectorSortedFiltered <- costSelectorSorted[mediaVarSortedFiltered]
  channelConstrLowSorted <- channel_constr_low[media_order][coefSelectorSorted]
  channelConstrUpSorted <- channel_constr_up[media_order][coefSelectorSorted]

  ## Get adstock parameters for each channel
  if (InputCollect$adstock == "geometric") {
    getAdstockHypPar <- unlist(dt_hyppar[, .SD, .SDcols = na.omit(str_extract(names(dt_hyppar), ".*_thetas"))])
  } else if (InputCollect$adstock %in% c("weibull_cdf", "weibull_pdf")) {
    getAdstockHypPar <- unlist(dt_hyppar[, .SD, .SDcols = na.omit(str_extract(names(dt_hyppar), ".*_shapes|.*_scales"))])
  }

  ## Get hill parameters for each channel
  hillHypParVec <- unlist(dt_hyppar[, .SD, .SDcols = na.omit(str_extract(names(dt_hyppar), ".*_alphas|.*_gammas"))])
  alphas <- hillHypParVec[str_which(names(hillHypParVec), "_alphas")]
  gammas <- hillHypParVec[str_which(names(hillHypParVec), "_gammas")]
  chnAdstocked <- OutputCollect$mediaVecCollect[
    type == "adstockedMedia" & solID == select_model, mediaVarSortedFiltered,
    with = FALSE][startRW:endRW]
  gammaTrans <- mapply(function(gamma, x) {
    round(quantile(seq(range(x)[1], range(x)[2], length.out = 100), gamma), 4)
  }, gamma = gammas, x = chnAdstocked)
  names(gammaTrans) <- names(gammas)
  coefs <- dt_coef[, coef]
  names(coefs) <- dt_coef[, rn]
  coefsFiltered <- coefs[mediaVarSortedFiltered]

  ## Build evaluation function
  if (any(InputCollect$costSelector)) {
    mm_lm_coefs <- spendExpoMod$coef_lm
    names(mm_lm_coefs) <- spendExpoMod$channel
  } else {
    mm_lm_coefs <- c()
  }

  ## Build constraints function with scenarios
  nPeriod <- nrow(dt_optimCost)
  xDecompAggMedia <- OutputCollect$xDecompAgg[
    solID == select_model & rn %in% InputCollect$paid_media_vars][order(rank(rn))]

  if ("max_historical_response" %in% scenario) {
    expected_spend <- sum(xDecompAggMedia$total_spend)
    expSpendUnitTotal <- sum(xDecompAggMedia$mean_spend) # expected_spend / nPeriod
  } else {
    expSpendUnitTotal <- expected_spend / (expected_spend_days / InputCollect$dayInterval)
  }

  # Gather all values that will be used internally on optim (nloptr)
  eval_list <- list(
    mm_lm_coefs = mm_lm_coefs,
    coefsFiltered = coefsFiltered,
    alphas = alphas,
    gammaTrans = gammaTrans,
    mediaVarSortedFiltered = mediaVarSortedFiltered,
    costSelectorSortedFiltered = costSelectorSortedFiltered,
    vmaxVec = vmaxVec,
    kmVec = kmVec,
    expSpendUnitTotal = expSpendUnitTotal)
  # So we can implicitly use these values within eval_f()
  # optim_env <- new.env(parent = globalenv())
  # optim_env$eval_list <- eval_list
  options("ROBYN_TEMP" = eval_list)

  # eval_f(c(1,1))
  # $objective
  # [1] -0.02318446
  # $gradient
  # [1] -1.923670e-06 -8.148831e-06 -3.163465e-02 -3.553371e-05
  # $objective.channel
  # [1] -6.590166e-07 -3.087475e-06 -2.316821e-02 -1.250144e-05

  histSpend <- xDecompAggMedia[, .(rn, total_spend)]
  histSpend <- histSpend$total_spend
  names(histSpend) <- sort(InputCollect$paid_media_vars)
  histSpendTotal <- sum(histSpend)
  histSpendUnitTotal <- sum(xDecompAggMedia$mean_spend) # histSpendTotal/ nPeriod
  histSpendUnit <- xDecompAggMedia[rn %in% mediaVarSortedFiltered, mean_spend]
  names(histSpendUnit) <- mediaVarSortedFiltered
  histSpendShare <- histSpendUnit/histSpendUnitTotal
  names(histSpendShare) <- mediaVarSortedFiltered

  # QA: check if objective function correctly implemented
  histResponseUnitModel <- setNames(
    xDecompAggMedia[rn %in% mediaVarSortedFiltered, get("mean_response")],
    mediaVarSortedFiltered)
  histResponseUnitAllocator <- unlist(-eval_f(histSpendUnit)[["objective.channel"]])
  identical(round(histResponseUnitModel, 3), round(histResponseUnitAllocator, 3))

  ## Set initial values and bounds
  x0 <- lb <- histSpendUnit * channelConstrLowSorted
  ub <- histSpendUnit * channelConstrUpSorted

  ## Set optim options
  if (optim_algo == "MMA_AUGLAG") {
    local_opts <- list(
      "algorithm" = "NLOPT_LD_MMA",
      "xtol_rel" = 1.0e-10
    )
  } else if (optim_algo == "SLSQP_AUGLAG") {
    local_opts <- list(
      "algorithm" = "NLOPT_LD_SLSQP",
      "xtol_rel" = 1.0e-10
    )
  }

  opts <- list(
    "algorithm" = "NLOPT_LD_AUGLAG",
    "xtol_rel" = 1.0e-10,
    "maxeval" = maxeval,
    "local_opts" = local_opts
  )

  ## Run optim
  nlsMod <- nloptr::nloptr(
    x0 = x0,
    eval_f = eval_f,
    eval_g_eq = if (constr_mode == "eq") eval_g_eq else NULL,
    eval_g_ineq = if (constr_mode == "ineq") eval_g_ineq else NULL,
    lb = lb, ub = ub,
    opts = opts)

  ## Collect output
  dt_bestModel <- dt_bestCoef[, .(rn, mean_spend, xDecompAgg, roi_total, roi_mean)][order(rank(rn))]
  dt_optimOut <- data.table(
    channels = mediaVarSortedFiltered,
    histSpend = histSpend[mediaVarSortedFiltered],
    histSpendTotal = histSpendTotal,
    initSpendUnitTotal = histSpendUnitTotal,
    initSpendUnit = histSpendUnit,
    initSpendShare = histSpendShare,
    initResponseUnit = histResponseUnitModel,
    initResponseUnitTotal = sum(xDecompAggMedia$mean_response),
    initRoiUnit = histResponseUnitModel / histSpendUnit,
    expSpendTotal = expected_spend,
    expSpendUnitTotal = expSpendUnitTotal,
    expSpendUnitDelta = expSpendUnitTotal / histSpendUnitTotal - 1,
    optmSpendUnit = nlsMod$solution,
    optmSpendUnitDelta = (nlsMod$solution / histSpendUnit - 1),
    optmSpendUnitTotal = sum(nlsMod$solution),
    optmSpendUnitTotalDelta = sum(nlsMod$solution) / histSpendUnitTotal - 1,
    optmSpendShareUnit = nlsMod$solution / sum(nlsMod$solution),
    optmResponseUnit = -eval_f(nlsMod$solution)[["objective.channel"]],
    optmResponseUnitTotal = sum(-eval_f(nlsMod$solution)[["objective.channel"]]),
    optmRoiUnit = -eval_f(nlsMod$solution)[["objective.channel"]] / nlsMod$solution,
    optmResponseUnitLift = (-eval_f(nlsMod$solution)[["objective.channel"]] / histResponseUnitModel) - 1
  )
  dt_optimOut[, optmResponseUnitTotalLift := (optmResponseUnitTotal / initResponseUnitTotal) - 1]
  .Options$ROBYN_TEMP <- NULL # Clean auxiliary method

  ## Plot allocator results
  plots <- allocation_plots(InputCollect, OutputCollect, dt_optimOut, select_model, export, quiet)

  ## Export results into CSV
  if (export) fwrite(dt_optimOut, paste0(OutputCollect$plot_folder, select_model, "_reallocated.csv"))

  return(list(
    dt_optimOut = dt_optimOut,
    nlsMod = nlsMod,
    ui = if (ui) plots else NULL))

}

eval_f <- function(X) {

  # eval_list <- get("eval_list", pos = as.environment(-1))
  eval_list <- getOption("ROBYN_TEMP")
  mm_lm_coefs <- eval_list[["mm_lm_coefs"]]
  coefsFiltered <- eval_list[["coefsFiltered"]]
  alphas <- eval_list[["alphas"]]
  gammaTrans <- eval_list[["gammaTrans"]]
  mediaVarSortedFiltered <- eval_list[["mediaVarSortedFiltered"]]
  costSelectorSortedFiltered <- eval_list[["costSelectorSortedFiltered"]]
  vmaxVec <- eval_list[["vmaxVec"]]
  kmVec <- eval_list[["kmVec"]]

  fx_objective <- function(x, coeff, alpha, gammaTran, chnName, vmax, km, criteria) {
    # Apply Michaelis Menten model to scale spend to exposure
    if (criteria) {
      xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
    } else if (chnName %in% names(mm_lm_coefs)) {
      xScaled <- x * mm_lm_coefs[chnName]
    } else {
      xScaled <- x
    }
    # Adstock scales
    xAdstocked <- xScaled
    # Hill transformation
    xOut <- coeff * sum((1 + gammaTran**alpha / xAdstocked**alpha)**-1)
    xOut
    return(xOut)
  }

  objective <- -sum(mapply(
    fx_objective,
    x = X,
    coeff = coefsFiltered,
    alpha = alphas,
    gammaTran = gammaTrans,
    chnName = mediaVarSortedFiltered,
    vmax = vmaxVec,
    km = kmVec,
    criteria = costSelectorSortedFiltered,
    SIMPLIFY = TRUE
  ))

  # https://www.derivative-calculator.net/ on the objective function 1/(1+gamma^alpha / x^alpha)
  fx_gradient <- function(x, coeff, alpha, gammaTran, chnName, vmax, km, criteria) {
    # Apply Michaelis Menten model to scale spend to exposure
    if (criteria) {
      xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
    } else if (chnName %in% names(mm_lm_coefs)) {
      xScaled <- x * mm_lm_coefs[chnName]
    } else {
      xScaled <- x
    }
    # Adstock scales
    xAdstocked <- xScaled
    xOut <- -coeff * sum((alpha * (gammaTran**alpha) * (xAdstocked**(alpha - 1))) / (xAdstocked**alpha + gammaTran**alpha)**2)
    return(xOut)
  }

  gradient <- c(mapply(
    fx_gradient,
    x = X,
    coeff = coefsFiltered,
    alpha = alphas,
    gammaTran = gammaTrans,
    chnName = mediaVarSortedFiltered,
    vmax = vmaxVec,
    km = kmVec,
    criteria = costSelectorSortedFiltered,
    SIMPLIFY = TRUE
  ))

  fx_objective.chanel <- function(x, coeff, alpha, gammaTran, chnName, vmax, km, criteria) {
    # Apply Michaelis Menten model to scale spend to exposure
    if (criteria) {
      xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
    } else if (chnName %in% names(mm_lm_coefs)) {
      xScaled <- x * mm_lm_coefs[chnName]
    } else {
      xScaled <- x
    }
    # Adstock scales
    xAdstocked <- xScaled
    xOut <- -coeff * sum((1 + gammaTran**alpha / xAdstocked**alpha)**-1)
    return(xOut)
  }

  objective.channel <- mapply(
    fx_objective.chanel,
    x = X,
    coeff = coefsFiltered,
    alpha = alphas,
    gammaTran = gammaTrans,
    chnName = mediaVarSortedFiltered,
    vmax = vmaxVec,
    km = kmVec,
    criteria = costSelectorSortedFiltered,
    SIMPLIFY = TRUE
  )

  optm <- list(objective = objective, gradient = gradient, objective.channel = objective.channel)
  return(optm)
}

eval_g_eq <- function(X) {
  eval_list <- getOption("ROBYN_TEMP")
  constr <- sum(X) - eval_list$expSpendUnitTotal
  grad <- rep(1, length(X))
  return(list(
    "constraints" = constr,
    "jacobian" = grad
  ))
}

eval_g_ineq <- function(X) {
  eval_list <- getOption("ROBYN_TEMP")
  constr <- sum(X) - eval_list$expSpendUnitTotal
  grad <- rep(1, length(X))
  return(list(
    "constraints" = constr,
    "jacobian" = grad
  ))
}
