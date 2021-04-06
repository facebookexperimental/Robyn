# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

################################################################
#### Define optimiser function

f.budgetAllocator <- function(modID = NULL
                              ,optim_algo = "MMA_AUGLAG"
                              ,expected_spend = NULL
                              ,expected_spend_days = NULL
                              ,channel_constr_low = 0.5
                              ,channel_constr_up = 2
                              ,scenario = "max_historical_response"
                              ,maxeval = 100000
                              ,constr_mode = "eq") {
  
  if (is.null(modID)) {
    stop("must provide modID, the model ID")
  }
  
  cat("\nRunning budget allocator for model ID", modID, "...\n")
  
  ## check input parameters
  if (any(channel_constr_low <0.01) | any(channel_constr_low >1)) {stop("channel_constr_low must be between 0.01 and 1")}
  if (any(channel_constr_up <1) | any(channel_constr_up >5)) {stop("channel_constr_up must be between 1-5")}
  if (!(scenario %in% c("max_historical_response", "max_response_expected_spend"))) {
    stop("scenario must be 'max_historical_response', 'max_response_expected_spend'")
  }
  
  if (length(channel_constr_up)!=1) {
    if (length(channel_constr_low)!= length(set_mediaVarName) | length(channel_constr_up)!= length(set_mediaVarName)) {
      stop("channel_constr_low & channel_constr_up have to contain either only 1 value or have same length as set_mediaVarName")
    }
  }
  
  ## get dt 
  dt_bestHyperParam <- model_output_collect$resultHypParam[solID == modID]
  if (!(modID %in% dt_bestHyperParam$solID)) {stop("provided modID is not within the best results")}
  dt_bestCoef <- model_output_collect$xDecompAgg[solID == modID & rn %in% set_mediaVarName]
  
  dt_spendShare <- dt_input[, .(rn = set_mediaVarName,
                                total_spend = sapply(.SD, sum)), .SDcols=set_mediaSpendName]
  dt_bestCoef[dt_spendShare[, .(rn, total_spend)], ':='(spend = i.total_spend, roi = xDecompAgg / i.total_spend), on = "rn"]
  dt_optim <- copy(dt_mod)
  
  ## get filter for channels mmm coef reduced to 0
  dt_coef <- dt_bestCoef[, .(rn, coef)]
  dt_coefSorted <- dt_coef[order(rn)]
  coefSelectorSorted <- dt_coefSorted[, coef>0]
  names(coefSelectorSorted) <- dt_coefSorted$rn
  
  ## filter and sort all variables by name that is essential for the apply function later
  channelNames <- sort(set_mediaVarName)
  channelNames <- channelNames[coefSelectorSorted]
  if(!all(coefSelectorSorted)) {
    chn_coef0 <- setdiff(set_mediaVarName, channelNames)
    message(paste(chn_coef0, collapse = ", "), " are excluded in optimiser because their coeffients are 0")
  }
  
  dt_bestHyperParam <- dt_bestHyperParam[, .SD, .SDcols = na.omit(str_extract(names(dt_bestHyperParam),paste(paste0(channelNames,".*"),collapse = "|")))]
  setcolorder(dt_bestHyperParam, sort(names(dt_bestHyperParam)))
  
  dt_optim <- dt_optim[, channelNames, with = F]
  setcolorder(dt_optim, sort(names(dt_optim)))
  
  dt_optimCost <- copy(dt_input)
  dt_optimCost <- dt_optimCost[, set_mediaSpendName, with = F]
  names(dt_optimCost) <- set_mediaVarName
  #dt_optimCost <- dt_optimCost[, channelNames, with=F]
  setcolorder(dt_optimCost, channelNames)
  
  dt_bestCoef <- dt_bestCoef[order(rn)][rn %in% channelNames]
  
  costMultiplierVec <- mediaCostFactor[channelNames]
  
  if(any(costSelector)) {
    dt_modNLS <- merge(data.table(channel=channelNames), modNLSCollect, all.x = T, by = "channel")
    vmaxVec <- dt_modNLS[order(channel)][, Vmax]
    names(vmaxVec) <- channelNames
    kmVec <- dt_modNLS[order(channel)][, Km]
    names(kmVec) <- channelNames
  } else {
    vmaxVec <- rep(0, length(channelNames))
    kmVec <- rep(0, length(channelNames))
  }
  
  #names(costSelector) <- set_mediaVarName
  costSelectorSorted <- costSelector[order(set_mediaVarName)]
  costSelectorSorted <- costSelectorSorted[coefSelectorSorted]
  costSelectorSorted <- costSelectorSorted[channelNames]
  
  names(channel_constr_low) <- set_mediaVarName; names(channel_constr_up) <- set_mediaVarName
  channelConstrLowSorted <- channel_constr_low[order(set_mediaVarName)][coefSelectorSorted]
  channelConstrUpSorted <- channel_constr_up[order(set_mediaVarName)][coefSelectorSorted]
  
  ## get adstock parameters for each channel
  if (adstock == "geometric") {
    getAdstockHypPar <- unlist(dt_bestHyperParam[, .SD, .SDcols = na.omit(str_extract(names(dt_bestHyperParam),".*_thetas"))])
  } else if (adstock == "weibull") {
    getAdstockHypPar <- unlist(dt_bestHyperParam[, .SD, .SDcols = na.omit(str_extract(names(dt_bestHyperParam),".*_shapes|.*_scales"))])
  }
  
  ## get hill parameters for each channel
  hillHypParVec <- unlist(dt_bestHyperParam[, .SD, .SDcols = na.omit(str_extract(names(dt_bestHyperParam),".*_alphas|.*_gammas"))])
  alphas <- hillHypParVec[str_which(names(hillHypParVec), "_alphas")]
  
  adstockTrans <- mapply(function(chnl) {
    
    alpha <- hillHypParVec[str_which(names(hillHypParVec), paste0(chnl, "_alphas"))]
    gamma <- hillHypParVec[str_which(names(hillHypParVec), paste0(chnl, "_gammas"))]
    chnRaw <- unlist(dt_optim[, chnl, with =F])
    
    if (adstock == "geometric") {
      chnAdstocked <- f.transformation(chnRaw
                                       ,theta = getAdstockHypPar[str_which(names(getAdstockHypPar), chnl)]
                                       ,alpha = alpha
                                       ,gamma = gamma
                                       ,alternative = adstock
                                       ,stage = 2)
    } else if (adstock == "weibull") {
      chnAdstocked <- f.transformation(chnRaw
                                       ,shape = getAdstockHypPar[str_which(names(getAdstockHypPar), paste0(chnl, "_shapes"))]
                                       ,scale = getAdstockHypPar[str_which(names(getAdstockHypPar), paste0(chnl, "_scales"))]
                                       ,alpha = alpha
                                       ,gamma = gamma
                                       ,alternative = adstock
                                       ,stage = 2)
    }
    
    gammaTrans <- round(quantile(seq(range(chnAdstocked)[1],range(chnAdstocked)[2], length.out = 100), gamma),4)
    names(gammaTrans) <- NULL
    
    return(list(gammaTrans=gammaTrans, chnAdstocked=chnAdstocked))
    
  }, chnl = channelNames, SIMPLIFY = F)
  
  gammaTrans <- sapply(adstockTrans, function(x) x[1][["gammaTrans"]])
  chnAdstocked <- as.data.table(sapply(mapply(function(lst, n) data.table(n=lst[2][["chnAdstocked"]]), lst = adstockTrans, n = names(adstockTrans)), cbind))
  names(chnAdstocked) <- channelNames
  adstockMultiplierVec <- colSums(chnAdstocked) / colSums(dt_optim)
  
  coefs <- dt_bestCoef[,coef]; names(coefs) <- dt_bestCoef[,rn]
  
  ## build evaluation funciton
  if(exists("modNLSCollect")) {
    mm_lm_coefs <- modNLSCollect$coef_lm
    names(mm_lm_coefs) <- modNLSCollect$channel
  } else {
    mm_lm_coefs <- c()
  }

  eval_f <- function(X) {
    return(
      list(
        "objective" = -sum(
          mapply(function(x, costMultiplier, adstockMultiplier, coeff
                          , alpha, gammaTran
                          , chnName, vmax, km, criteria) {
            # apply Michaelis Menten model to scale spend to reach
            if (criteria) {
              xScaled <- vmax * x / (km + x)
            } else if (chnName %in% names(mm_lm_coefs)) {
                xScaled <- x * mm_lm_coefs[chnName]
            } else {
              xScaled <- x 
            }
            
            # adstock scales
            xAdstocked <- xScaled#* adstockMultiplier
            
            # hill transformation
            #xOut <- coeff * sum( (1 + gammaTran**alpha / (x/costMultiplier*adstockMultiplier) **alpha)**-1) 
            xOut <- coeff * sum( (1 + gammaTran**alpha / xAdstocked **alpha)**-1)
            
            return(xOut)
          }, x=X, costMultiplier = costMultiplierVec, adstockMultiplier=adstockMultiplierVec, coeff = coefs
          , alpha = alphas, gammaTran = gammaTrans
          , chnName = channelNames
          , vmax = vmaxVec, km = kmVec, criteria = costSelectorSorted
          , SIMPLIFY = T)
        ),
        
        "gradient" = c(
          mapply(function(x, costMultiplier, adstockMultiplier, coeff
                          , alpha, gammaTran
                          , chnName, vmax, km, criteria) {
            # apply Michaelis Menten model to scale spend to reach
            if (criteria) {
              xScaled <- vmax * x / (km + x)
            } else if (chnName %in% names(mm_lm_coefs)) {
              xScaled <- x * mm_lm_coefs[chnName]
            } else {
              xScaled <- x 
            }
            
            # adstock scales
            xAdstocked <- xScaled#*adstockMultiplier
            
            #xOut <- -coeff * sum((alpha * (gammaTran**alpha) * ((x/costMultiplier*adstockMultiplier)**(alpha-1))) / ((x/costMultiplierVec*adstockMultiplierVec)**alpha + gammaTran**alpha)**2)
            xOut <- -coeff * sum((alpha * (gammaTran**alpha) * (xAdstocked**(alpha-1))) / (xAdstocked**alpha + gammaTran**alpha)**2)
            
            return(xOut)
          }, x=X, costMultiplier = costMultiplierVec, adstockMultiplier=adstockMultiplierVec, coeff = coefs
          , alpha = alphas, gammaTran = gammaTrans
          , chnName = channelNames
          , vmax = vmaxVec, km = kmVec, criteria = costSelectorSorted
          , SIMPLIFY = T)
        ), # https://www.derivative-calculator.net/ on the objective function 1/(1+gamma^alpha / x^alpha)
        
        "objective.channel" =
          mapply(function(x, costMultiplier, adstockMultiplier, coeff
                          , alpha, gammaTran
                          , chnName, vmax, km, criteria) {
            
            # apply Michaelis Menten model to scale spend to reach
            if (criteria) {
              xScaled <- vmax * x / (km + x)
            } else if (chnName %in% names(mm_lm_coefs)) {
              xScaled <- x * mm_lm_coefs[chnName]
            } else {
              xScaled <- x 
            }
            
            # adstock scales
            xAdstocked <- xScaled#*adstockMultiplier
            
            #xOut <- -coeff * sum( (1 + gammaTran**alpha / (x/costMultiplier*adstockMultiplier) **alpha)**-1)
            xOut <- -coeff * sum( (1 + gammaTran**alpha / xAdstocked **alpha)**-1)
            
            return(xOut)
          }, x=X, costMultiplier = costMultiplierVec, adstockMultiplier=adstockMultiplierVec, coeff = coefs
          , alpha = alphas, gammaTran = gammaTrans
          , chnName = channelNames
          , vmax = vmaxVec, km = kmVec, criteria = costSelectorSorted
          , SIMPLIFY = T)
        
      ))}
  
  #eval_f(c(1,1))
  
  ## build contraints function with scenarios
  nPeriod <- nrow(dt_optimCost)
  xDecompAggMedia <- model_output_collect$xDecompAgg[solID==modID & rn %in% set_mediaVarName][order(rn)]
  
  if (scenario == "max_historical_response") {
    expected_spend <- sum(xDecompAggMedia$total_spend)
    expSpendUnitTotal <- sum(xDecompAggMedia$mean_spend)  #expected_spend / nPeriod
    
  } else if (scenario == "max_response_expected_spend") {
    
    if (any(is.null(expected_spend), is.null(expected_spend_days))) {
      stop("when scenario = 'max_response_expected_spend', expected_spend and expected_spend_days must be provided")
    }
    expSpendUnitTotal <- expected_spend / (expected_spend_days / dayInterval)
  }
  
  histSpend <- xDecompAggMedia[,.(rn, total_spend)]
  histSpend <- histSpend$total_spend; names(histSpend) <- sort(set_mediaVarName)
  #histSpend <- colSums(dt_optimCost)
  histSpendTotal <- sum(histSpend)
  histSpendUnitTotal <- sum(xDecompAggMedia$mean_spend) # histSpendTotal/ nPeriod
  #histSpendShare <- histSpend / histSpendTotal
  #histSpendUnit <- histSpendUnitTotal * histSpendShare
  histSpendUnit <- xDecompAggMedia[rn %in% channelNames, mean_spend]; names(histSpendUnit) <- channelNames
  histSpendShare <- xDecompAggMedia[rn %in% channelNames, spend_share]; names(histSpendShare) <- channelNames
  
  # QA: check if objective function correctly implemented
  histResponseUnitModel <- xDecompAggMedia[rn %in% channelNames, get("mean_response")]; names(histResponseUnitModel) <- channelNames
  histResponseUnitAllocator <- unlist(-eval_f(histSpendUnit)[["objective.channel"]])
  identical(round(histResponseUnitModel,3), round(histResponseUnitAllocator,3))
  
  # for (i in 1:length(chn_coef0)) {
  #   histResponseUnit[length(channelNames)+i] <- 0
  #   names(histResponseUnit)[length(channelNames)+i] <- chn_coef0[i]
  # }
  # histResponseUnit <- histResponseUnit[names(histSpend)]
  
  # expSpendUnit <- histSpendShare * expSpendUnitTotal
  # expResponseUnit <- -eval_f(expSpendUnit)[["objective.channel"]]
  
  eval_g_eq <- function(X) {
    constr <- sum(X) - expSpendUnitTotal
    grad <- rep(1, length(X))
    return( list("constraints" = constr
                 ,"jacobian" = grad))
  }
  
  eval_g_ineq <- function(X) {
    constr <- sum(X) - expSpendUnitTotal
    grad <- rep(1, length(X))
    return( list("constraints" = constr
                 ,"jacobian" = grad))
  }
  
  ## set initial values and bounds
  lb <- histSpendUnit * channelConstrLowSorted
  ub <- histSpendUnit * channelConstrUpSorted
  x0 <- lb
  
  ## set optim options
  if (optim_algo == "MMA_AUGLAG") {
    local_opts <- list( "algorithm" = "NLOPT_LD_MMA",
                        "xtol_rel" = 1.0e-10 )
    opts <- list( "algorithm" = "NLOPT_LD_AUGLAG",
                  "xtol_rel" = 1.0e-10,
                  "maxeval" = maxeval,
                  "local_opts" = local_opts )
  }
  
  ## run optim
  if (constr_mode  == "eq") {
    nlsMod <- nloptr( x0=x0,
                      eval_f=eval_f,
                      lb=lb,
                      ub=ub,
                      #eval_g_ineq=eval_g_ineq,
                      eval_g_eq=eval_g_eq,
                      opts=opts)
  } else if (constr_mode  == "ineq") {
    nlsMod <- nloptr( x0=x0,
                      eval_f=eval_f,
                      lb=lb,
                      ub=ub,
                      eval_g_ineq=eval_g_ineq,
                      #eval_g_eq=eval_g_eq,
                      opts=opts)
  }
  
  #print(nlsMod)
  
  
  ## collect output 
  
  dt_bestModel <- dt_bestCoef[, .(rn, spend, xDecompAgg, roi)][order(rn)]
  
  dt_optimOut <- data.table(
    channels = channelNames
    ,histSpend = histSpend[channelNames]
    ,histSpendTotal = histSpendTotal
    ,initSpendUnitTotal = histSpendUnitTotal
    ,initSpendUnit = histSpendUnit
    ,initSpendShare = histSpendShare
    # ,histResponse = dt_bestModel$xDecompAgg
    # ,histResponseTotal = sum(dt_bestModel$xDecompAgg)
    # ,histResponseUnit = dt_bestModel$xDecompAgg/nPeriod
    # ,histResponseUnitTotal = sum(dt_bestModel$xDecompAgg/nPeriod)
    ,initResponseUnit = histResponseUnitModel
    ,initResponseUnitTotal = sum(xDecompAggMedia$mean_response)
    ,initRoiUnit = histResponseUnitModel/histSpendUnit
    ,expSpendTotal = expected_spend
    ,expSpendUnitTotal = expSpendUnitTotal
    ,expSpendUnitDelta = expSpendUnitTotal/histSpendUnitTotal-1
    #,expSpendUnit = expSpendUnit
    # ,expResponseUnit = expResponseUnit
    # ,expResponseUnitTotal = sum(expResponseUnit)
    # ,expRoiUnit = expResponseUnit / expSpendUnit
    ,optmSpendUnit = nlsMod$solution
    ,optmSpendUnitDelta = (nlsMod$solution / histSpendUnit -1)
    ,optmSpendUnitTotal = sum(nlsMod$solution)
    ,optmSpendUnitTotalDelta = sum(nlsMod$solution)/histSpendUnitTotal-1
    ,optmSpendShareUnit = nlsMod$solution / sum(nlsMod$solution)
    ,optmResponseUnit = -eval_f(nlsMod$solution)[["objective.channel"]]
    ,optmResponseUnitTotal = sum(-eval_f(nlsMod$solution)[["objective.channel"]])
    ,optmRoiUnit = -eval_f(nlsMod$solution)[["objective.channel"]] / nlsMod$solution
    ,optmResponseUnitLift = (-eval_f(nlsMod$solution)[["objective.channel"]] / histResponseUnitModel) -1
  )
  
  dt_optimOut[, optmResponseUnitTotalLift:= (optmResponseUnitTotal / initResponseUnitTotal) -1]
  print(dt_optimOut)
  
  ## plot allocator results
  
  plotDT_total <- copy(dt_optimOut) # plotDT_total <- optim_result$dt_optimOut
  
  # ROI comparison plot
  
  plotDT_roi <- plotDT_total[, c("channels", "initRoiUnit", "optmRoiUnit")][order(channels)]
  plotDT_roi[, channels:=as.factor(channels)]
  chn_levels <- plotDT_roi[, as.character(channels)]
  plotDT_roi[, channels:=factor(channels, levels = chn_levels)]
  setnames(plotDT_roi, names(plotDT_roi), new = c("channel", "initial roi", "optimised roi"))
  
  
  plotDT_roi <- suppressWarnings(melt.data.table(plotDT_roi, id.vars = "channel", value.name = "roi"))
  p11 <- ggplot(plotDT_roi, aes(x=channel, y=roi,fill = variable)) +
          geom_bar(stat = "identity", width = 0.5, position = "dodge") +
          coord_flip() +
          scale_fill_brewer(palette = "Paired") +
          geom_text(aes(label=round(roi,2), hjust=1, size=2.0),  position=position_dodge(width=0.5), fontface = "bold", show.legend = FALSE) +
          theme( legend.title = element_blank(), legend.position = c(0.9, 0.2) ,axis.text.x = element_blank(), legend.background=element_rect(colour='grey', fill='transparent')) +
          labs(title = "Initial vs. optimised mean ROI"
               ,subtitle = paste0("Total spend increases ", plotDT_total[, round(mean(optmSpendUnitTotalDelta)*100,1)], "%"
                                  ,"\nTotal response increases ", plotDT_total[, round(mean(optmResponseUnitTotalLift)*100,1)], "% with optimised spend allocation")
               ,y="", x="Channels")
  
  # Response comparison plot
  plotDT_resp <- plotDT_total[, c("channels", "initResponseUnit", "optmResponseUnit")][order(channels)]
  plotDT_resp[, channels:=as.factor(channels)]
  chn_levels <- plotDT_resp[, as.character(channels)]
  plotDT_resp[, channels:=factor(channels, levels = chn_levels)]
  setnames(plotDT_resp, names(plotDT_resp), new = c("channel", "initial response / time unit", "optimised response / time unit"))
  
  plotDT_resp <- suppressWarnings(melt.data.table(plotDT_resp, id.vars = "channel", value.name = "response"))
  p12 <- ggplot(plotDT_resp, aes(x=channel, y=response, fill = variable)) +
          geom_bar(stat = "identity", width = 0.5, position = "dodge") +
          coord_flip() +
          scale_fill_brewer(palette = "Paired") +
          geom_text(aes(label=round(response,0), hjust=1, size=2.0), position=position_dodge(width=0.5), fontface = "bold", show.legend = FALSE) +
          theme( legend.title = element_blank(), legend.position = c(0.8, 0.2) ,axis.text.x = element_blank(), legend.background=element_rect(colour='grey', fill='transparent')) +
          labs(title = "Initial vs. optimised mean response"
               ,subtitle = paste0("Total spend increases ", plotDT_total[, round(mean(optmSpendUnitTotalDelta)*100,1)], "%"
                                  ,"\nTotal response increases ", plotDT_total[, round(mean(optmResponseUnitTotalLift)*100,1)], "% with optimised spend allocation"
               )
               ,y="", x="Channels")
  
  # budget share comparison plot
  plotDT_share <- plotDT_total[, c("channels", "initSpendShare", "optmSpendShareUnit")][order(channels)]
  plotDT_share[, channels:=as.factor(channels)]
  chn_levels <- plotDT_share[, as.character(channels)]
  plotDT_share[, channels:=factor(channels, levels = chn_levels)]
  setnames(plotDT_share, names(plotDT_share), new = c("channel", "initial avg.spend share", "optimised avg.spend share"))
  
  plotDT_share <- suppressWarnings(melt.data.table(plotDT_share, id.vars = "channel", value.name = "spend_share"))
  p13 <- ggplot(plotDT_share, aes(x=channel, y=spend_share, fill = variable)) +
    geom_bar(stat = "identity", width = 0.5, position = "dodge") +
    coord_flip() +
    scale_fill_brewer(palette = "Paired") +
    geom_text(aes(label=paste0(round(spend_share*100,2),"%"), hjust=1, size=2.0), position=position_dodge(width=0.5), fontface = "bold", show.legend = FALSE) +
    theme( legend.title = element_blank(), legend.position = c(0.8, 0.2) ,axis.text.x = element_blank(), legend.background=element_rect(colour='grey', fill='transparent')) +
    labs(title = "Initial vs. optimised budget allocation"
         ,subtitle = paste0("Total spend increases ", plotDT_total[, round(mean(optmSpendUnitTotalDelta)*100,1)], "%"
                            ,"\nTotal response increases ", plotDT_total[, round(mean(optmResponseUnitTotalLift)*100,1)], "% with optimised spend allocation"
         )
         ,y="", x="Channels")
  
  
  ## response curve
  
  plotDT_saturation <- melt.data.table(model_output_collect$mediaVecCollect[solID==modID & type == "saturatedSpendReversed"], id.vars = "ds", measure.vars = set_mediaVarName, value.name = "spend", variable.name = "channel")
  plotDT_decomp <- melt.data.table(model_output_collect$mediaVecCollect[solID==modID & type == "decompMedia"], id.vars = "ds", measure.vars = set_mediaVarName, value.name = "response", variable.name = "channel")
  plotDT_scurve <- cbind(plotDT_saturation, plotDT_decomp[, .(response)])
  plotDT_scurve <- plotDT_scurve[spend>=0] # remove outlier introduced by MM nls fitting
  plotDT_scurveMeanResponse <- model_output_collect$xDecompAgg[solID==modID & rn %in% set_mediaVarName]
  dt_optimOutScurve <- rbind(dt_optimOut[, .(channels, initSpendUnit, initResponseUnit)][, type:="initial"], dt_optimOut[, .(channels, optmSpendUnit, optmResponseUnit)][, type:="optimised"], use.names = F)
  setnames(dt_optimOutScurve, c("channels", "spend", "response", "type"))
  
  p14 <- ggplot(data= plotDT_scurve, aes(x=spend, y=response, color = channel)) +
    geom_line() +
    geom_point(data = dt_optimOutScurve, aes(x=spend, y=response, color = channels, shape = type), size = 2) +
    geom_text(data = dt_optimOutScurve, aes(x=spend, y=response, color = channels, label = round(spend,0)),  show.legend = F, hjust = -0.2) +
    #geom_point(data = dt_optimOut, aes(x=optmSpendUnit, y=optmResponseUnit, color = channels, fill = "optimised"), shape=2) +
    #geom_text(data = dt_optimOut, aes(x=optmSpendUnit, y=optmResponseUnit, color = channels, label = round(optmSpendUnit,0)),  show.legend = F, hjust = -0.2) +
    theme(legend.position = c(0.9, 0.4), legend.title=element_blank()) +
    labs(title="Response curve and mean spend by channel"
         ,subtitle = paste0("rsq_test: ", plotDT_scurveMeanResponse[,round(mean(rsq_test),4)], 
                            ", nrmse = ", plotDT_scurveMeanResponse[, round(mean(nrmse),4)], 
                            ", decomp.rssd = ", plotDT_scurveMeanResponse[, round(mean(decomp.rssd),4)],
                            ", mape.lift = ", plotDT_scurveMeanResponse[, round(mean(mape),4)])
         ,x="Spend" ,y="response")
  
  
  grobTitle <- paste0("Budget allocator optimum result for model ID ", modID)
  
  # pgbl <- arrangeGrob(p13,p12,p11,p14, ncol=2, top = text_grob(grobTitle, size = 15, face = "bold"))
  # grid.draw(pgbl)
  
  g13 <- ggplotGrob(p13)
  g12 <- ggplotGrob(p12)
  g14 <- ggplotGrob(p14)
  maxWidth <- unit.pmax(g13$widths, g12$widths, g14$widths)
  g13$widths <- g12$widths <- g14$widths <- maxWidth
  layout <- cbind(c(1,2), c(3,3))
  g <- grid.arrange(g13, g12, g14,   layout_matrix=layout, top = text_grob(grobTitle, size = 15, face = "bold"))
  
  cat("\nSaving plots to ", paste0(model_output_collect$folder_path, modID,"_reallocated.png"), "...\n")
  ggsave(filename=paste0(model_output_collect$folder_path, modID,"_reallocated.png")
         , plot = g
         , dpi = 400, width = 18, height = 14)
  
  fwrite(dt_optimOut, paste0(model_output_collect$folder_path, modID,"_reallocated.csv"))
  return(list(dt_optimOut=dt_optimOut, nlsMod=nlsMod))
}


#lib https://cran.r-project.org/web/packages/nloptr/nloptr.pdf non linear function with equal and unequal constraints + bounds
#find gradient of a function https://socratic.org/questions/how-do-you-find-the-gradient-of-a-function-at-a-given-point

# # lib https://cran.r-project.org/web/packages/nloptr/nloptr.pdf
# # find gradient of a function https://socratic.org/questions/how-do-you-find-the-gradient-of-a-function-at-a-given-point
