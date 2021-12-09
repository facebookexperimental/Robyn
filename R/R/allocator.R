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
#' @param robyn_object Character. Path of the \code{Robyn.RDS} object
#' that contains all previous modeling information.
#' @param select_build Integer. Default to the latest model build. \code{select_buil = 0}
#' selects the initial model. \code{select_buil = 1} selects the first refresh model.
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
                            ui = FALSE) {

  #####################################
  #### Set local environment

  ## collect input
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
        "Using latest model: ", ifelse(select_build == 0, "initial model", paste0(
          "refresh model nr.", select_build
        )),
        " for the response function. select_build = 0 selects initial model, 1 the first refresh etc"
      )
    }

    if (!(select_build %in% select_build_all) | length(select_build) != 1) {
      stop("select_build must be one value of ", paste(select_build_all, collapse = ", "))
    }

    listName <- ifelse(select_build == 0, "listInit", paste0("listRefresh", select_build))
    InputCollect <- Robyn[[listName]][["InputCollect"]]
    OutputCollect <- Robyn[[listName]][["OutputCollect"]]
    select_model <- OutputCollect$selectID
  } else if (any(is.null(InputCollect), is.null(OutputCollect), is.null(select_model))) {
    stop("when robyn_object is not provided, then InputCollect, OutputCollect, select_model must be provided")
  }

  message(paste(">>> Running budget allocator for model ID", select_model, "..."))

  ## get data & params
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

  dt_hyppar <- OutputCollect$resultHypParam[solID == select_model]
  if (!(select_model %in% dt_hyppar$solID)) {
    stop("provided select_model is not within the best results")
  }

  dt_bestCoef <- OutputCollect$xDecompAgg[solID == select_model & rn %in% InputCollect$paid_media_vars]

  ## check input parameters

  if (any(channel_constr_low < 0.01)) {
    stop("'channel_constr_low' must be >= 0.01")
  }
  if (any(channel_constr_up < channel_constr_low)) {
    stop("'channel_constr_up' must be >= 'channel_constr_low'")
  }
  if (any(channel_constr_up > 5)) {
    message("'channel_constr_up' >5 might cause unrealistic allocation")
  }
  if (!(scenario %in% c("max_historical_response", "max_response_expected_spend"))) {
    stop("scenario must be 'max_historical_response', 'max_response_expected_spend'")
  }

  if (length(channel_constr_up) != 1) {
    if (length(channel_constr_low) != length(InputCollect$paid_media_vars) |
      length(channel_constr_up) != length(InputCollect$paid_media_vars)) {
      stop(paste(
        "'channel_constr_low' & 'channel_constr_up' have to contain either only 1",
        "value or have same length as 'InputCollect$paid_media_vars'"
      ))
    }
  }

  names(channel_constr_low) <- paid_media_vars
  names(channel_constr_up) <- paid_media_vars

  ## filter and sort

  dt_mediaSpend <- dt_input[startRW:endRW, mediaSpendSorted, with = FALSE]

  ## sort table and get filter for channels mmm coef reduced to 0
  dt_coef <- dt_bestCoef[, .(rn, coef)]
  get_rn_order <- order(dt_bestCoef$rn)
  dt_coefSorted <- dt_coef[get_rn_order]
  dt_bestCoef <- dt_bestCoef[get_rn_order]
  coefSelectorSorted <- dt_coefSorted[, coef > 0]
  names(coefSelectorSorted) <- dt_coefSorted$rn

  ## filter and sort all variables by name that is essential for the apply function later
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

  ## get adstock parameters for each channel
  if (InputCollect$adstock == "geometric") {
    getAdstockHypPar <- unlist(dt_hyppar[, .SD, .SDcols = na.omit(str_extract(names(dt_hyppar), ".*_thetas"))])
  } else if (InputCollect$adstock %in% c("weibull_cdf", "weibull_pdf")) {
    getAdstockHypPar <- unlist(dt_hyppar[, .SD, .SDcols = na.omit(str_extract(names(dt_hyppar), ".*_shapes|.*_scales"))])
  }

  ## get hill parameters for each channel
  hillHypParVec <- unlist(dt_hyppar[, .SD, .SDcols = na.omit(str_extract(names(dt_hyppar), ".*_alphas|.*_gammas"))])
  alphas <- hillHypParVec[str_which(names(hillHypParVec), "_alphas")]
  gammas <- hillHypParVec[str_which(names(hillHypParVec), "_gammas")]

  chnAdstocked <- OutputCollect$mediaVecCollect[
    type == "adstockedMedia" & solID == select_model, mediaVarSortedFiltered,
    with = FALSE
  ][startRW:endRW]
  gammaTrans <- mapply(function(gamma, x) {
    round(quantile(seq(range(x)[1], range(x)[2], length.out = 100), gamma), 4)
  },
  gamma = gammas,
  x = chnAdstocked
  )
  names(gammaTrans) <- names(gammas)

  coefs <- dt_coef[, coef]
  names(coefs) <- dt_coef[, rn]
  coefsFiltered <- coefs[mediaVarSortedFiltered]

  ## build evaluation funciton
  if (any(InputCollect$costSelector)) {
    mm_lm_coefs <- spendExpoMod$coef_lm
    names(mm_lm_coefs) <- spendExpoMod$channel
  } else {
    mm_lm_coefs <- c()
  }

  # sl=4;coeff = coefsFiltered[sl]; alpha = alphas[sl]; gammaTran = gammaTrans[sl]; chnName = mediaVarSortedFiltered[sl]; vmax = vmaxVec[sl]; km = kmVec[sl]; criteria = costSelectorSortedFiltered[sl]
  # coeff* saturation_hill(x=chnAdstocked[, get(chnName)], alpha = alpha, gamma = gammas[sl], x_marginal = mic_men(x=256198.38, Vmax=vmax, Km=km))
  # coeff* saturation_hill(x=chnAdstocked[, get(chnName)], alpha = alpha, gamma = gammas[sl], x_marginal =257771.9)

  eval_f <- function(X) {
    return(
      list(
        "objective" = -sum(
          mapply(function(x # , costMultiplier, adstockMultiplier
                          , coeff,
                          alpha, gammaTran,
                          chnName, vmax, km, criteria) {
            # apply Michaelis Menten model to scale spend to exposure
            if (criteria) {
              xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
            } else if (chnName %in% names(mm_lm_coefs)) {
              xScaled <- x * mm_lm_coefs[chnName]
            } else {
              xScaled <- x
            }

            # adstock scales
            xAdstocked <- xScaled #* adstockMultiplier

            # hill transformation
            # xOut <- coeff * sum( (1 + gammaTran**alpha / (x/costMultiplier*adstockMultiplier) **alpha)**-1)
            xOut <- coeff * sum((1 + gammaTran**alpha / xAdstocked**alpha)**-1)
            xOut


            return(xOut)
          },
          x = X # , costMultiplier = costMultiplierVec, adstockMultiplier=adstockMultiplierVec
          , coeff = coefsFiltered,
          alpha = alphas, gammaTran = gammaTrans,
          chnName = mediaVarSortedFiltered,
          vmax = vmaxVec, km = kmVec, criteria = costSelectorSortedFiltered,
          SIMPLIFY = TRUE
          )
        ),
        "gradient" = c(
          mapply(function(x # , costMultiplier, adstockMultiplier
                          , coeff,
                          alpha, gammaTran,
                          chnName, vmax, km, criteria) {
            # apply Michaelis Menten model to scale spend to exposure
            if (criteria) {
              xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
            } else if (chnName %in% names(mm_lm_coefs)) {
              xScaled <- x * mm_lm_coefs[chnName]
            } else {
              xScaled <- x
            }

            # adstock scales
            xAdstocked <- xScaled #* adstockMultiplier

            xOut <- -coeff * sum((alpha * (gammaTran**alpha) * (xAdstocked**(alpha - 1))) /
              (xAdstocked**alpha + gammaTran**alpha)**2)

            return(xOut)
          },
          x = X # , costMultiplier = costMultiplierVec, adstockMultiplier=adstockMultiplierVec
          , coeff = coefsFiltered,
          alpha = alphas, gammaTran = gammaTrans,
          chnName = mediaVarSortedFiltered,
          vmax = vmaxVec, km = kmVec, criteria = costSelectorSortedFiltered,
          SIMPLIFY = TRUE
          )
        ), # https://www.derivative-calculator.net/ on the objective function 1/(1+gamma^alpha / x^alpha)

        "objective.channel" =
          mapply(function(x # , costMultiplier, adstockMultiplier
                          , coeff,
                          alpha, gammaTran,
                          chnName, vmax, km, criteria) {

            # apply Michaelis Menten model to scale spend to exposure
            if (criteria) {
              xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
            } else if (chnName %in% names(mm_lm_coefs)) {
              xScaled <- x * mm_lm_coefs[chnName]
            } else {
              xScaled <- x
            }

            # adstock scales
            xAdstocked <- xScaled #* adstockMultiplier

            # xOut <- -coeff * sum( (1 + gammaTran**alpha / (x/costMultiplier*adstockMultiplier) **alpha)**-1)
            xOut <- -coeff * sum((1 + gammaTran**alpha / xAdstocked**alpha)**-1)

            return(xOut)
          },
          x = X # , costMultiplier = costMultiplierVec, adstockMultiplier=adstockMultiplierVec
          , coeff = coefsFiltered,
          alpha = alphas, gammaTran = gammaTrans,
          chnName = mediaVarSortedFiltered,
          vmax = vmaxVec, km = kmVec, criteria = costSelectorSortedFiltered,
          SIMPLIFY = TRUE
          )
      )
    )
  }

  # eval_f(c(1,1))

  ## build contraints function with scenarios
  nPeriod <- nrow(dt_optimCost)
  xDecompAggMedia <- OutputCollect$xDecompAgg[solID == select_model & rn %in% InputCollect$paid_media_vars][order(rank(rn))]

  if (scenario == "max_historical_response") {
    expected_spend <- sum(xDecompAggMedia$total_spend)
    expSpendUnitTotal <- sum(xDecompAggMedia$mean_spend) # expected_spend / nPeriod
  } else if (scenario == "max_response_expected_spend") {
    if (any(is.null(expected_spend), is.null(expected_spend_days))) {
      stop("when scenario = 'max_response_expected_spend', expected_spend and expected_spend_days must be provided")
    }
    expSpendUnitTotal <- expected_spend / (expected_spend_days / InputCollect$dayInterval)
  }

  histSpend <- xDecompAggMedia[, .(rn, total_spend)]
  histSpend <- histSpend$total_spend
  names(histSpend) <- sort(InputCollect$paid_media_vars)
  # histSpend <- colSums(dt_optimCost)
  histSpendTotal <- sum(histSpend)
  histSpendUnitTotal <- sum(xDecompAggMedia$mean_spend) # histSpendTotal/ nPeriod
  # histSpendShare <- histSpend / histSpendTotal
  # histSpendUnit <- histSpendUnitTotal * histSpendShare
  histSpendUnit <- xDecompAggMedia[rn %in% mediaVarSortedFiltered, mean_spend]
  names(histSpendUnit) <- mediaVarSortedFiltered
  #histSpendShare <- xDecompAggMedia[rn %in% mediaVarSortedFiltered, spend_share]
  histSpendShare <- histSpendUnit/histSpendUnitTotal
  names(histSpendShare) <- mediaVarSortedFiltered

  # QA: check if objective function correctly implemented
  histResponseUnitModel <- xDecompAggMedia[rn %in% mediaVarSortedFiltered, get("mean_response")]
  names(histResponseUnitModel) <- mediaVarSortedFiltered
  histResponseUnitAllocator <- unlist(-eval_f(histSpendUnit)[["objective.channel"]])
  identical(round(histResponseUnitModel, 3), round(histResponseUnitAllocator, 3))

  eval_g_eq <- function(X) {
    constr <- sum(X) - expSpendUnitTotal
    grad <- rep(1, length(X))
    return(list(
      "constraints" = constr,
      "jacobian" = grad
    ))
  }

  eval_g_ineq <- function(X) {
    constr <- sum(X) - expSpendUnitTotal
    grad <- rep(1, length(X))
    return(list(
      "constraints" = constr,
      "jacobian" = grad
    ))
  }

  ## set initial values and bounds
  lb <- histSpendUnit * channelConstrLowSorted
  ub <- histSpendUnit * channelConstrUpSorted
  x0 <- lb

  ## set optim options
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

  ## run optim
  if (constr_mode == "eq") {
    nlsMod <- nloptr::nloptr(
      x0 = x0,
      eval_f = eval_f,
      lb = lb,
      ub = ub,
      # eval_g_ineq=eval_g_ineq,
      eval_g_eq = eval_g_eq,
      opts = opts
    )
  } else if (constr_mode == "ineq") {
    nlsMod <- nloptr::nloptr(
      x0 = x0,
      eval_f = eval_f,
      lb = lb,
      ub = ub,
      eval_g_ineq = eval_g_ineq,
      # eval_g_eq=eval_g_eq,
      opts = opts
    )
  }

  # print(nlsMod)

  ## collect output

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
  # print(dt_optimOut)

  ## plot allocator results

  plotDT_total <- copy(dt_optimOut) # plotDT_total <- optim_result$dt_optimOut

  # ROI comparison plot

  plotDT_roi <- plotDT_total[, c("channels", "initRoiUnit", "optmRoiUnit")][order(rank(channels))]
  plotDT_roi[, channels := as.factor(channels)]
  chn_levels <- plotDT_roi[, as.character(channels)]
  plotDT_roi[, channels := factor(channels, levels = chn_levels)]
  setnames(plotDT_roi, names(plotDT_roi), new = c("channel", "initial roi", "optimised roi"))


  plotDT_roi <- suppressWarnings(melt.data.table(plotDT_roi, id.vars = "channel", value.name = "roi"))
  p11 <- ggplot(plotDT_roi, aes(x = channel, y = roi, fill = variable)) +
    geom_bar(stat = "identity", width = 0.5, position = "dodge") +
    coord_flip() +
    scale_fill_brewer(palette = "Paired") +
    geom_text(aes(label = round(roi, 2), hjust = 1, size = 2.0),
      position = position_dodge(width = 0.5), fontface = "bold", show.legend = FALSE
    ) +
    theme(
      legend.title = element_blank(), legend.position = c(0.9, 0.2),
      axis.text.x = element_blank(), legend.background = element_rect(
        colour = "grey", fill = "transparent"
      )
    ) +
    labs(
      title = "Initial vs. optimised mean ROI",
      subtitle = paste0(
        "Total spend increases ", plotDT_total[
          , round(mean(optmSpendUnitTotalDelta) * 100, 1)
        ], "%",
        "\nTotal response increases ", plotDT_total[
          , round(mean(optmResponseUnitTotalLift) * 100, 1)
        ], "% with optimised spend allocation"
      ),
      y = "", x = "Channels"
    )

  # Response comparison plot
  plotDT_resp <- plotDT_total[, c("channels", "initResponseUnit", "optmResponseUnit")][order(rank(channels))]
  plotDT_resp[, channels := as.factor(channels)]
  chn_levels <- plotDT_resp[, as.character(channels)]
  plotDT_resp[, channels := factor(channels, levels = chn_levels)]
  setnames(plotDT_resp, names(plotDT_resp), new = c("channel", "initial response / time unit", "optimised response / time unit"))

  plotDT_resp <- suppressWarnings(melt.data.table(plotDT_resp, id.vars = "channel", value.name = "response"))
  p12 <- ggplot(plotDT_resp, aes(x = channel, y = response, fill = variable)) +
    geom_bar(stat = "identity", width = 0.5, position = "dodge") +
    coord_flip() +
    scale_fill_brewer(palette = "Paired") +
    geom_text(aes(label = round(response, 0), hjust = 1, size = 2.0),
      position = position_dodge(width = 0.5), fontface = "bold", show.legend = FALSE
    ) +
    theme(
      legend.title = element_blank(), legend.position = c(0.8, 0.2),
      axis.text.x = element_blank(), legend.background = element_rect(
        colour = "grey", fill = "transparent"
      )
    ) +
    labs(
      title = "Initial vs. optimised mean response",
      subtitle = paste0(
        "Total spend increases ", plotDT_total[
          , round(mean(optmSpendUnitTotalDelta) * 100, 1)
        ], "%",
        "\nTotal response increases ", plotDT_total[
          , round(mean(optmResponseUnitTotalLift) * 100, 1)
        ], "% with optimised spend allocation"
      ),
      y = "", x = "Channels"
    )

  # budget share comparison plot
  plotDT_share <- plotDT_total[, c("channels", "initSpendShare", "optmSpendShareUnit")][order(rank(channels))]
  plotDT_share[, channels := as.factor(channels)]
  chn_levels <- plotDT_share[, as.character(channels)]
  plotDT_share[, channels := factor(channels, levels = chn_levels)]
  setnames(plotDT_share, names(plotDT_share), new = c("channel", "initial avg.spend share", "optimised avg.spend share"))

  plotDT_share <- suppressWarnings(melt.data.table(plotDT_share, id.vars = "channel", value.name = "spend_share"))
  p13 <- ggplot(plotDT_share, aes(x = channel, y = spend_share, fill = variable)) +
    geom_bar(stat = "identity", width = 0.5, position = "dodge") +
    coord_flip() +
    scale_fill_brewer(palette = "Paired") +
    geom_text(aes(label = paste0(round(spend_share * 100, 2), "%"), hjust = 1, size = 2.0),
      position = position_dodge(width = 0.5), fontface = "bold", show.legend = FALSE
    ) +
    theme(
      legend.title = element_blank(), legend.position = c(0.8, 0.2),
      axis.text.x = element_blank(), legend.background = element_rect(
        colour = "grey", fill = "transparent"
      )
    ) +
    labs(
      title = "Initial vs. optimised budget allocation",
      subtitle = paste0(
        "Total spend increases ", plotDT_total[, round(mean(optmSpendUnitTotalDelta) * 100, 1)], "%",
        "\nTotal response increases ", plotDT_total[, round(mean(optmResponseUnitTotalLift) * 100, 1)], "% with optimised spend allocation"
      ),
      y = "", x = "Channels"
    )


  ## response curve

  plotDT_saturation <- melt.data.table(OutputCollect$mediaVecCollect[
    solID == select_model & type == "saturatedSpendReversed"
  ],
  id.vars = "ds",
  measure.vars = InputCollect$paid_media_vars, value.name = "spend", variable.name = "channel"
  )
  plotDT_decomp <- melt.data.table(OutputCollect$mediaVecCollect[
    solID == select_model & type == "decompMedia"
  ],
  id.vars = "ds",
  measure.vars = InputCollect$paid_media_vars, value.name = "response", variable.name = "channel"
  )
  plotDT_scurve <- cbind(plotDT_saturation, plotDT_decomp[, .(response)])
  plotDT_scurve <- plotDT_scurve[spend >= 0] # remove outlier introduced by MM nls fitting
  plotDT_scurveMeanResponse <- OutputCollect$xDecompAgg[solID == select_model & rn %in% InputCollect$paid_media_vars]
  dt_optimOutScurve <- rbind(dt_optimOut[
    , .(channels, initSpendUnit, initResponseUnit)
  ][, type := "initial"],
  dt_optimOut[, .(channels, optmSpendUnit, optmResponseUnit)][, type := "optimised"],
  use.names = FALSE
  )
  setnames(dt_optimOutScurve, c("channels", "spend", "response", "type"))

  p14 <- ggplot(data = plotDT_scurve, aes(x = spend, y = response, color = channel)) +
    geom_line() +
    geom_point(data = dt_optimOutScurve, aes(
      x = spend, y = response, color = channels, shape = type
    ), size = 2) +
    geom_text(
      data = dt_optimOutScurve, aes(
        x = spend, y = response, color = channels, label = round(spend, 0)
      ),
      show.legend = FALSE, hjust = -0.2
    ) +
    theme(legend.position = c(0.9, 0.4), legend.title = element_blank()) +
    labs(
      title = "Response curve and mean spend by channel",
      subtitle = paste0(
        "rsq_train: ", plotDT_scurveMeanResponse[, round(mean(rsq_train), 4)],
        ", nrmse = ", plotDT_scurveMeanResponse[, round(mean(nrmse), 4)],
        ", decomp.rssd = ", plotDT_scurveMeanResponse[, round(mean(decomp.rssd), 4)],
        ", mape.lift = ", plotDT_scurveMeanResponse[, round(mean(mape), 4)]
      ),
      x = "Spend", y = "response"
    )

  grobTitle <- paste0("Budget allocator optimum result for model ID ", select_model)
  g <- (p13 + p12) / p14 + plot_annotation(
    title = grobTitle, theme = theme(plot.title = element_text(hjust = 0.5))
  )

  message("Exporting charts into file: ", paste0(OutputCollect$plot_folder, select_model, "_reallocated.png"))
  ggsave(
    filename = paste0(OutputCollect$plot_folder, select_model, "_reallocated.png"),
    plot = g,
    dpi = 400, width = 18, height = 14
  )

  fwrite(dt_optimOut, paste0(OutputCollect$plot_folder, select_model, "_reallocated.csv"))

  if (ui) {
    ui <- list(p12 = p12, p13 = p13, p14 = p14)
  } else {
    ui <- NULL
  }

  listAllocator <- list(dt_optimOut = dt_optimOut, nlsMod = nlsMod, ui = ui)

  return(listAllocator)
}
