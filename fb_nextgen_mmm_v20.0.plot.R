# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

################################################################
#### Define training size guidance plot using Bhattacharyya coefficient

f.plotTrainSize <- function(plotTrainSize) {
  
  if(plotTrainSize) {
    if(activate_baseline & exists("set_baseVarName")) {
      bhattaVar <- unique(c(set_depVarName, set_baseVarName, set_mediaVarName, set_mediaSpendName))
    } else {stop("either set activate_baseline = F or fill set_baseVarName")}
    bhattaVar <- setdiff(bhattaVar, set_factorVarName)
    dt_bhatta <- dt_input[, bhattaVar, with=F]  # please input your data
    
    ## define bhattacharyya distance function
    f.bhattaCoef <- function (mu1, mu2, Sigma1, Sigma2) {
      Sig <- (Sigma1 + Sigma2)/2
      ldet.s <- unlist(determinant(Sig, logarithm = TRUE))[1]
      ldet.s1 <- unlist(determinant(Sigma1, logarithm = TRUE))[1]
      ldet.s2 <- unlist(determinant(Sigma2, logarithm = TRUE))[1]
      d1 <- mahalanobis(mu1, mu2, Sig, tol=1e-20)/8
      d2 <- 0.5 * ldet.s - 0.25 * ldet.s1 - 0.25 * ldet.s2
      d <- d1 + d2
      bhatta.coef <- 1/exp(d)
      return(bhatta.coef)
    }
    
    ## loop all train sizes
    bcCollect <- c()
    sizeVec <- seq(from=0.5, to=0.9, by=0.01)
    
    for (i in 1:length(sizeVec)) {
      test1 <- dt_bhatta[1:floor(nrow(dt_bhatta)*sizeVec[i]), ]
      test2 <- dt_bhatta[(floor(nrow(dt_bhatta)*sizeVec[i])+1):nrow(dt_bhatta), ]
      bcCollect[i] <- f.bhattaCoef(colMeans(test1),colMeans(test2),cov(test1),cov(test2)) 
    }
    
    dt_bdPlot <- data.table(train_size=sizeVec, bhatta_coef=bcCollect)
    
    print(ggplot(dt_bdPlot, aes(x=train_size, y=bhatta_coef)) + 
            geom_line() +
            labs(title = "Bhattacharyya coef. of train/test split"
                 ,subtitle = "Select the training size with larger bhatta_coef"))
  }
}

f.plotAdstockCurves <- function(plotAdstockCurves) {
  if (plotAdstockCurves) {
    if (adstock == "weibull") {
      weibullCollect <- list()
      shapeVec <- c(2, 2, 2, 2, 2, 2, 0.01, 0.1, 0.5, 1, 1.5, 2)
      scaleVec <- c(0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
      paramRotate <- c(rep("scale",6), rep("shape",6))
      
      for (v in 1:length(shapeVec)) {
        dt_weibull<- data.table(x=1:100,
                                decay_accumulated=f.adstockWeibull(rep(1, 100), shape=shapeVec[v], scale=scaleVec[v])$thetaVecCum
                                ,shape=shapeVec[v]
                                ,scale=scaleVec[v]
                                ,param=paramRotate[v])
        dt_weibull[, halflife:= which.min(abs(decay_accumulated -0.5))]
        
        weibullCollect[[v]] <- dt_weibull
      }
      
      weibullCollect <- rbindlist(weibullCollect)
      #weibullCollect[, ':='(shape=as.factor(shape), scale=as.factor(scale))]
      weibullCollect[, scale_shape_halflife:=paste(scale, shape,halflife, sep = "_")]
      p1 <- ggplot(weibullCollect[param=="scale"], aes(x=x, y=decay_accumulated)) + 
        geom_line(aes(color=scale_shape_halflife)) +
        geom_hline(yintercept=0.5, linetype="dashed", color = "gray") +
        geom_text(aes(x = max(x), y = 0.5, vjust = -0.5, hjust= 1, label = "Halflife = time until effect reduces to 50%"), colour="gray") +
        labs(title="Weibull adstock transformation - scale changes", 
             subtitle="Halflife = time until effect reduces to 50%",
             x="time unit",
             y="Media decay accumulated") 
      p2 <- ggplot(weibullCollect[param=="shape"], aes(x=x, y=decay_accumulated)) + 
        geom_line(aes(color=scale_shape_halflife)) +
        geom_hline(yintercept=0.5, linetype="dashed", color = "gray") +
        geom_text(aes(x = max(x), y = 0.5, vjust = -0.5, hjust= 1, label = "Halflife = time until effect reduces to 50%"), colour="gray") +
        labs(title="Weibull adstock transformation - shape changes", 
             subtitle="Halflife = time until effect reduces to 50%",
             x="time unit",
             y="Media decay accumulated") 
      
      grid.arrange(p1,p2)
      
    } else if (adstock == "geometric") {
      
      geomCollect <- list()
      thetaVec <- c(0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9)
      
      for (v in 1:length(thetaVec)) {
        thetaVecCum <- 1
        for (t in 2:100) {thetaVecCum[t] <- thetaVecCum[t-1]*thetaVec[v]}
        dt_geom <- data.table(x=1:100,
                              decay_accumulated = thetaVecCum,
                              theta=thetaVec[v])
        dt_geom[, halflife:= which.min(abs(decay_accumulated -0.5))]
        geomCollect[[v]] <- dt_geom
      }
      geomCollect <- rbindlist(geomCollect)
      geomCollect[, theta_halflife:=paste(theta,halflife, sep = "_")]
      
      p3 <- ggplot(geomCollect, aes(x=x, y=decay_accumulated)) + 
        geom_line(aes(color=theta_halflife)) +
        geom_hline(yintercept=0.5, linetype="dashed", color = "gray") +
        geom_text(aes(x = max(x), y = 0.5, vjust = -0.5, hjust= 1, label = "Halflife = time until effect reduces to 50%"), colour="gray") +
        labs(title="Geometric adstock transformation - theta changes", 
             subtitle="Halflife = time until effect reduces to 50%",
             x="time unit",
             y="Media decay accumulated") 
      print(p3)
    }
  }
}

f.plotResponseCurves <- function(plotResponseCurves, alphaFix = 2, gammaFix = 0.5) {
  if (plotResponseCurves) {
    xSample <- 1:100
    alphaSamp <- c(0.1, 0.5, 1, 2, 3)
    gammaSamp <- c(0.1, 0.3, 0.5, 0.7, 0.9)
    
    ## plot alphas
    hillAlphaCollect <- list()
    for (i in 1:length(alphaSamp)) {
      hillAlphaCollect[[i]] <- data.table(x=xSample
                                          ,y=xSample**alphaSamp[i] / (xSample**alphaSamp[i] + (gammaFix*100)**alphaSamp[i])
                                          ,alpha=alphaSamp[i])
    }
    hillAlphaCollect <- rbindlist(hillAlphaCollect)
    hillAlphaCollect[, alpha:=as.factor(alpha)]
    p1 <- ggplot(hillAlphaCollect, aes(x=x, y=y, color=alpha)) + 
      geom_line() +
      labs(title = "Cost response with hill function"
           ,subtitle = paste0("Alpha changes while gamma = ", gammaFix))
    
    ## plot gammas
    hillGammaCollect <- list()
    for (i in 1:length(gammaSamp)) {
      hillGammaCollect[[i]] <- data.table(x=xSample
                                          ,y=xSample**alphaFix / (xSample**alphaFix + (gammaSamp[i]*100)**alphaFix)
                                          ,gamma=gammaSamp[i])
    }
    hillGammaCollect <- rbindlist(hillGammaCollect)
    hillGammaCollect[, gamma:=as.factor(gamma)]
    p2 <- ggplot(hillGammaCollect, aes(x=x, y=y, color=gamma)) + 
      geom_line() +
      labs(title = "Cost response with hill function"
           ,subtitle = paste0("Gamma changes while alpha = ", alphaFix))
    
    grid.arrange(p1,p2, nrow=1)
  }
}


#####################################
#### Define prophet plot function

f.plotTrendSeason <- function(plotTrendSeason=T)  {
  if (plotTrendSeason & activate_prophet) {
    prophet_plot_components(modelRecurrance, forecastRecurrance)
  }
}

#####################################
#### Define media transformation plot function, including 3 plots: decay plot, adstock plot and response curve plot

f.plotMediaTransform <- function(plotMediaTransform, channelPlot = NULL) {
  if (plotMediaTransform) {
    
    best_iter <- best_model$resultCollect$best.iter
    plotDT <- best_model
    
    best.rsq <- plotDT$resultCollect$resultHypParam[iterRS == best_iter, rsq_test]
    best.mape <- plotDT$resultCollect$resultHypParam[iterRS == best_iter, mape]
    hypParam <- unlist(plotDT$resultCollect$resultHypParam[iterRS == best_iter, !c("lambdas", "mape", "rsq_test", "pos", "Score", "Elapsed", "ElapsedAccum", "iterRS"), with =F])
    
    ## filter max. 3 channels at one time
    if (is.null(channelPlot)) {
      set_mediaVarNamePlot <- na.omit(set_mediaVarName[1:3])
      #colSelect <- str_detect(names(dt_mod), paste0(c("ds",set_mediaVarName[1:3]), collapse = "|"))
    } else {
      if (length(channelPlot)>3) {message("input should have max.3 channels. plotting the first 3 now.")}
      #set_mediaVarNamePlot <- na.omit(toupper(channelPlot)[1:3])
      set_mediaVarNamePlot <- na.omit(channelPlot[1:3])
      
      #colSelect <- str_detect(names(dt_mod), paste0(c("ds", channelPlot[1:3]), collapse = "|"))
    }
    colSelect <- c("ds", set_mediaVarNamePlot)
    adstockDT <- dt_mod[, colSelect, with =F]
    
    ## adstock decay vector
    m_decayRate <- list()
    if (adstock == "geometric") {
      for (med in 1:length(set_mediaVarNamePlot)) {
        m <- adstockDT[, get(set_mediaVarNamePlot[med])]
        theta <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_thetas")==names(hypParam))]
        alpha <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_alphas")==names(hypParam))]
        gamma <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_gammas")==names(hypParam))]
        m_decayRate[[med]] <- data.table((f.transformation(x=m, theta=theta, alpha=alpha, gamma=gamma, alternative = adstock, stage="thetaVecCum")))
        setnames(m_decayRate[[med]], "V1", paste0(set_mediaVarNamePlot[med], "_decayRate"))
      }
      subVal <- hypParam[str_detect(names(hypParam), paste0("(",paste(set_mediaVarNamePlot, collapse = "|"),").*", "theta"))]
      subT <- paste(paste0(names(subVal), ": ", round(subVal,4)), collapse = "|")
    } else if (adstock == "weibull") {
      for (med in 1:length(set_mediaVarNamePlot)) {
        m <- adstockDT[, get(set_mediaVarNamePlot[med])]
        shape <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_shapes")==names(hypParam))]
        scale <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_scales")==names(hypParam))]
        alpha <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_alphas")==names(hypParam))]
        gamma <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_gammas")==names(hypParam))]
        m_decayRate[[med]] <- data.table((f.transformation(x=m, shape= shape, scale=scale, alpha=alpha, gamma=gamma, alternative = adstock, stage="thetaVecCum")))
        setnames(m_decayRate[[med]], "V1", paste0(set_mediaVarNamePlot[med], "_decayRate"))
      }
      subVal <- hypParam[str_detect(names(hypParam), paste0("(",paste(set_mediaVarNamePlot, collapse = "|"),").*", "(shape|scale)") )]
      subT <- paste(paste0(names(subVal), ": ", round(subVal,4)), collapse = "|")
    }
    
    m_decayRate <- data.table(cbind(sapply(m_decayRate, function(x) sapply(x, function(y)y))))
    setnames(m_decayRate, names(m_decayRate), paste0(set_mediaVarNamePlot, "_decayRate"))
    m_decayRate[, ds:=adstockDT$ds]
    
    decayRate.melt <- suppressWarnings(melt.data.table(m_decayRate, id.vars = "ds"))
    decayRate.melt[, channel:=str_extract(decayRate.melt$variable, paste0(set_mediaVarNamePlot, collapse = "|"))]
    decayRate.melt[, variable:=str_replace(decayRate.melt$variable, paste0(paste0(set_mediaVarNamePlot,"_"), collapse = "|"), "")]
    
    print(ggplot(decayRate.melt, aes(x=ds, y=value, color=variable)) +
            geom_line()+
            facet_grid(channel ~., scales = "free") +
            labs(title=paste("Media adstocking",adstock), 
                 subtitle=subT,
                 x="ds",
                 y="Media decay accumulated") 
    )
    
    ## adstock before & after
    m_decayed <- list()
    if (adstock == "geometric") {
      for (med in 1:length(set_mediaVarNamePlot)) {
        m <- adstockDT[, get(set_mediaVarNamePlot[med])]
        theta <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_thetas")==names(hypParam))]
        alpha <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_alphas")==names(hypParam))]
        gamma <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_gammas")==names(hypParam))]
        m_decayed[[med]] <- data.table((f.transformation(x=m, theta=theta, alpha=alpha, gamma=gamma, alternative = adstock, stage=1)))
        setnames(m_decayed[[med]], "V1", paste0(set_mediaVarNamePlot[med], "_decayed"))
      }
    } else if (adstock == "weibull") {
      for (med in 1:length(set_mediaVarNamePlot)) {
        m <- adstockDT[, get(set_mediaVarNamePlot[med])]
        shape <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_shapes")==names(hypParam))]
        scale <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_scales")==names(hypParam))]
        alpha <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_alphas")==names(hypParam))]
        gamma <- hypParam[which(paste0(set_mediaVarNamePlot[med], "_gammas")==names(hypParam))]
        m_decayed[[med]] <- data.table((f.transformation(x=m, shape= shape, scale=scale, alpha=alpha, gamma=gamma, alternative = adstock, stage=1)))
        setnames(m_decayed[[med]], "V1", paste0(set_mediaVarNamePlot[med], "_decayed"))
      }
    }
    m_decayed <- data.table(cbind(sapply(m_decayed, function(x) sapply(x, function(y)y))))
    setnames(m_decayed, names(m_decayed), paste0(set_mediaVarNamePlot, "_decayed"))
    adstockDT_append <- cbind(adstockDT, m_decayed)
    
    setnames(adstockDT_append, set_mediaVarNamePlot, paste0(set_mediaVarNamePlot, "_original"))
    adstockDT.melt <- suppressWarnings(melt.data.table(adstockDT_append, id.vars = "ds"))
    adstockDT.melt[, channel:=str_extract(adstockDT.melt$variable, paste0(set_mediaVarNamePlot, collapse = "|"))]
    adstockDT.melt[, variable:=str_replace(adstockDT.melt$variable, paste0(paste0(set_mediaVarNamePlot,"_"), collapse = "|"), "")]
    
    print(ggplot(adstockDT.melt, aes(x=ds, y=value, color=variable)) +
            geom_line()+
            facet_grid(channel ~., scales = "free") +
            labs(title=paste("Media adstocking",adstock), 
                 subtitle=subT,
                 x="ds",
                 y="Response") 
    )
    
    ## plot response curve
    
    areaDT <- plotDT$resultCollect$xDecompVec[iterRS == best_iter, !c("mape", "iterRS", "y", "y_pred", "rsq_test")]
    areaDT <- areaDT[, baseline:= rowSums(.SD), .SDcols = c("intercept", set_baseVarName)][, !c("intercept", set_baseVarName), with = F]
    suppressWarnings({areaDT.melt <- melt.data.table(areaDT, id.vars = c("ds"), value.name = "fitted")})
    dt_plotResponse <- areaDT.melt[variable != "baseline"]
    suppressWarnings({dt_plotResponse <- dt_plotResponse[melt.data.table(dt_mod[, c("ds", set_mediaVarName), with =F], id.vars = c("ds"), value.name = "rawInput"), on = c("ds", "variable")]})#[, afterTheta:= 0]
    
    dt_agg <- plotDT$resultCollect$xDecompAgg[iterRS==best_iter, .(rn, coef, roi)]
    names(dt_agg) <- c("rn", "coef", "avg_roi")
    
    collectPlotScurve <- list()
    for (chn in set_mediaVarNamePlot) {
      if (any(names(plotDT$resultCollect$resultHypParam) %like% "theta")) {
        getTheta <- unlist(plotDT$resultCollect$resultHypParam[iterRS == best_iter, paste0(chn, "_", "thetas"), with = F])
        getAlpha <- unlist(plotDT$resultCollect$resultHypParam[iterRS == best_iter, paste0(chn, "_", "alphas"), with = F])
        getGamma <- unlist(plotDT$resultCollect$resultHypParam[iterRS == best_iter, paste0(chn, "_", "gammas"), with = F])
        dt_plotResponse[variable== chn, ':='(x_adstocked = f.transformation(x = rawInput
                                                                            , theta = getTheta
                                                                            , alpha = getAlpha
                                                                            , gamma = getGamma
                                                                            , alternative = "geometric"
                                                                            , stage = 1)
                                             ,x_scurve = f.transformation(x = rawInput
                                                                          , theta = getTheta
                                                                          , alpha = getAlpha
                                                                          , gamma = getGamma
                                                                          , alternative = "geometric"
                                                                          , stage = 3))]
        
        if (chn %in% set_mediaVarName[costSelector]) {
          yhatNLSCollect[channel== chn & models == "nls", ':='(x_adstocked_nls = f.transformation(x = yhat
                                                                                                  , theta = getTheta
                                                                                                  , alpha = getAlpha
                                                                                                  , gamma = getGamma
                                                                                                  , alternative = "geometric"
                                                                                                  , stage = 1)
                                                               ,x_scurve_nls = f.transformation(x = yhat
                                                                                                , theta = getTheta
                                                                                                , alpha = getAlpha
                                                                                                , gamma = getGamma
                                                                                                , alternative = "geometric"
                                                                                                , stage = 3))]
        }
        
      } else {
        getShape <- unlist(plotDT$resultCollect$resultHypParam[iterRS == best_iter, paste0(chn, "_", "shapes"), with = F])
        getScale <- unlist(plotDT$resultCollect$resultHypParam[iterRS == best_iter, paste0(chn, "_", "scales"), with = F])
        getAlpha <- unlist(plotDT$resultCollect$resultHypParam[iterRS == best_iter, paste0(chn, "_", "alphas"), with = F])
        getGamma <- unlist(plotDT$resultCollect$resultHypParam[iterRS == best_iter, paste0(chn, "_", "gammas"), with = F])
        dt_plotResponse[variable== chn, ':='(x_adstocked = f.transformation(x = rawInput
                                                                            , shape = getShape
                                                                            , scale = getScale
                                                                            , alpha = getAlpha
                                                                            , gamma = getGamma
                                                                            , alternative = "weibull"
                                                                            , stage = 1)
                                             ,x_scurve = f.transformation(x = rawInput
                                                                          , shape = getShape
                                                                          , scale = getScale
                                                                          , alpha = getAlpha
                                                                          , gamma = getGamma
                                                                          , alternative = "weibull"
                                                                          , stage = 3))]
        
        if (chn %in% set_mediaVarName[costSelector]) {
          yhatNLSCollect[channel== chn & models == "nls", ':='(x_adstocked_nls = f.transformation(x = yhat
                                                                                                  , shape = getShape
                                                                                                  , scale = getScale
                                                                                                  , alpha = getAlpha
                                                                                                  , gamma = getGamma
                                                                                                  , alternative = "weibull"
                                                                                                  , stage = 1)
                                                               ,x_scurve_nls = f.transformation(x = yhat
                                                                                                , shape = getShape
                                                                                                , scale = getScale
                                                                                                , alpha = getAlpha
                                                                                                , gamma = getGamma
                                                                                                , alternative = "weibull"
                                                                                                , stage = 3))]
        }
        
      }
      
      if (chn %in% set_mediaVarName[costSelector]) {
        toPlot <- cbind(dt_plotResponse[variable== chn],yhatNLSCollect[channel==chn & models == "nls"])
      } else {
        toPlot <- dt_plotResponse[variable== chn]
      }
      
      toPlot <- merge(toPlot, dt_agg, all.x = T, by.x = "variable", by.y = "rn")
      toPlot[, x_scurve_decomp := x_scurve * coef]
      mediaCostFactorPlot <- mediaCostFactor[chn]
      mediaAdstockFactorPlot <- toPlot[, sum(rawInput)/sum(x_adstocked)]
      toPlot[, x_adstocked_spendScaled:=x_adstocked*mediaAdstockFactorPlot] # scale adstocked reach back to raw reach level
      
      if (chn %in% set_mediaVarName[costSelector]) {
        Vmax <- modNLSCollect[channel == chn, Vmax]
        Km <- modNLSCollect[channel == chn, Km]
        toPlot[, x_adstocked_spendScaled:=x_adstocked_spendScaled * Km / (Vmax - x_adstocked_spendScaled)] # reach to spend, reverse Michaelis Menthen: x = y*Km/(Vmax-y)
        spendRatioFitted <- toPlot[, plotDT$resultCollect$xDecompAgg[rn == chn, sum(spend)]/sum(x_adstocked_spendScaled)]
        toPlot[, x_adstocked_spendScaled:= x_adstocked_spendScaled * spendRatioFitted] # scale reversed spend to true spend level
      } 
      #toPlot[, x_adstocked_spend:=x_adstocked * mediaCostFactorPlot]
      #secAxisScale <- toPlot[,quantile(x_scurve_decomp,0.6)/quantile(roi, 0.5)] 
      # toPlot[, x_adstocked_spendScaled:=x_adstocked_spend*mediaAdstockFactorPlot]
      # toPlot[, x_adstocked_spendScaled:=x_adstocked_spend]
      
      toPlot[, profit:=x_scurve_decomp - x_adstocked_spendScaled]
      toPlot[, roi := ifelse(is.na(x_scurve_decomp /x_adstocked_spendScaled), 0, x_scurve_decomp /x_adstocked_spendScaled)]
      #toPlot[, roi := roi / max(roi)] # indexed roi
      
      nVarPlot <- ifelse(length(set_mediaVarNamePlot)<=3,length(set_mediaVarNamePlot), 3)
      
      toPlotResponse <- toPlot[, .(x_adstocked_spendScaled, x_scurve_decomp, profit)]
      names(toPlotResponse) <- c("spend", "response", "profit (y - x)")
      toPlotResponse <- melt.data.table(toPlotResponse, id.vars = "spend")
      
      if (set_depVarType == "revenue") {
        collectPlotScurve[[which(set_mediaVarNamePlot == chn)]] <- ggplot(data=toPlotResponse,aes(x=spend, y=value, color = variable)) +
          geom_line() +
          geom_vline(data = toPlot, aes(xintercept = x_adstocked_spendScaled[which.max(profit)]), colour="cornflowerblue", linetype = "dashed") +
          geom_text(data = toPlot, aes(x = x_adstocked_spendScaled[which.max(profit)], y = max(x_scurve_decomp), angle = 90, vjust = -0.5, hjust= 1, label = paste0("spend lvl on max. profit: ",round(x_adstocked_spendScaled[which.max(profit)],0))), colour="steelblue") +
          labs(subtitle=paste0(chn, " response: \nalpha = ",round(getAlpha,2)," / gamma = ",round(getGamma,2)), y = "response", x = "spend") +
          theme(legend.position = c(0.85, 0.15), legend.title = element_blank(), legend.background = element_rect(fill=alpha('white', 0.5)))
      } else if (set_depVarType == "conversion") {
        collectPlotScurve[[which(set_mediaVarNamePlot == chn)]] <- ggplot(data=toPlot, aes(x=x_adstocked_spendScaled, y=x_scurve_decomp)) +
          geom_line(colour = "coral") +
          labs(subtitle=paste0(chn, " response: \nalpha = ",round(getAlpha,2)," / gamma = ",round(getGamma,2)), y = "response", x = "spend")
      }
      
      avgROI <- dt_agg[rn==chn, round(avg_roi,2)]
      
      collectPlotScurve[[which(set_mediaVarNamePlot == chn)+nVarPlot]] <- ggplot(data=toPlot, aes(x=x_adstocked_spendScaled, y=roi)) +
        geom_line(color = "gray30") +
        geom_vline(aes(xintercept = x_adstocked_spendScaled[which.max(roi)]), colour="steelblue", linetype = "dashed") +
        geom_text(aes(x = x_adstocked_spendScaled[which.max(roi)], y = max(roi), angle = 90, vjust = -0.5, hjust= 1, label = paste0("spend lvl on max.ROI: ",round(x_adstocked_spendScaled[which.max(roi)],0))), colour="steelblue")+
        geom_hline(yintercept=avgROI, linetype="dashed", color = "steelblue") + 
        geom_text(aes(x = max(x_adstocked_spendScaled), y = mean(avg_roi), vjust = -0.5, hjust= 1, label = paste0("avg_roi: ",round(mean(avg_roi),2))), colour="steelblue") +
        labs(subtitle=paste0(chn, " ROI:", "\navg_roi = ", avgROI, " (response/spend)"), y = "ROI", x = "spend")
    }
    
    grid.arrange(grobs = collectPlotScurve
                 ,ncol=nVarPlot
                 ,top = "Cost-response & indexed ROI curve per channel")
    
    return(list(decayRate = decayRate.melt, adstockDT = adstockDT.melt))
  }
}

#####################################
#### Define helper unit format function for axis 

f.unit_format <- function(x_in) {
  x_out <- sapply(x_in, function(x) {
    if (abs(x) >= 1000000000) {
      x_out <- paste0(round(x/1000000000, 1), " bln")
    } else if (abs(x) >= 1000000 & abs(x)<1000000000) {
      x_out <- paste0(round(x/1000000, 1), " mio")
    } else if (abs(x) >= 1000 & abs(x)<1000000) {
      x_out <- paste0(round(x/1000, 1), " tsd")
    } else {
      x_out <- round(x,0)
    }
  }, simplify = T) 
  return(x_out)
}

#####################################
#### Plotting model decomp results, incl. 3 plots: decomp waterfall plot, actual vs. fitted line plot and decomp area plot

f.plotBestDecomp <- function(plotBestDecomp=T) {
  if (plotBestDecomp) {
    best_iter <- best_model$resultCollect$best.iter
    plotDT <- best_model
    
    
    best.rsq <- plotDT$resultCollect$xDecompAgg[iterRS == best_iter, rsq_test]
    best.mape <- plotDT$resultCollect$xDecompAgg[iterRS == best_iter, mape]
    
    ## Bar chart for decomp
    
    barDT <- plotDT$resultCollect$xDecompAgg[iterRS == best_iter, !c("mape", "iterRS", "rsq_test", "pos"), with =F][order(-xDecompPerc)]
    #barDT <- barDT[ , rn := mapply( function(x,y) paste0(x, "_" ,y) , x = nrow(barDT):1, y =  paste0(rn, ": ",round(xDecompPerc *100, 1),"%"))][order(-xDecompPerc)]
    barDT[, end := cumsum(xDecompPerc)]
    barDT[, ':='(start =shift(end, fill = 0, type = "lag")
                 ,id = 1:nrow(barDT)
                 ,rn = as.factor(rn)
                 ,sign = as.factor(ifelse(xDecompPerc>=0, "pos", "neg")))]
    
    suppressWarnings(print(ggplot(barDT, aes(x= id, fill = sign)) +
                             geom_rect(aes(x = rn, xmin = id - 0.45, xmax = id + 0.45, ymin = end, ymax = start), stat="identity") +
                             scale_x_discrete("", breaks = levels(barDT$rn), labels = barDT$rn)+
                             theme(axis.text.x = element_text(angle=65, vjust=0.6))  +
                             geom_text(mapping = aes(label = paste0(f.unit_format(xDecompAgg),"\n", round(xDecompPerc*100, 2), "%"),y = rowSums(cbind(start,xDecompPerc/2))), fontface = "bold") +
                             labs(title="Response decomposition waterfall by predictor", 
                                  subtitle=paste0("rsq = ", round(best.rsq,4), " , mape = ", round(best.mape,4) ),
                                  x="Predictors",
                                  y="Decomp%")
    ))
    
    ## Line chart for actual vs fitted
    lineDT <- plotDT$resultCollect$xDecompVec[iterRS == best_iter]
    suppressWarnings(lineDT.melt <- melt.data.table(lineDT[, .(ds, y, y_pred)], id.vars = "ds"))
    dateIntercept <- lineDT[floor(set_modTrainSize* nrow(lineDT)), ds]
    print(ggplot(aes(y = value, x = ds, colour = variable), data = lineDT.melt) + 
            geom_line(stat="identity") +
            geom_vline(aes(xintercept = dateIntercept), colour="gray", linetype= "dashed") +
            geom_text(data = lineDT, aes(x = dateIntercept, y = max(y), angle = 90, vjust = -0.5, hjust= 1, label = paste0("training size: ",round(set_modTrainSize*100),"%")), colour="gray") +
            labs(title="Sales actual vs. fitted", 
                 subtitle=paste0("test.rsq = ", round(best.rsq,4), " , test.mape = ", round(best.mape,4) ),
                 x="ds",
                 y="Sales"))
    
    ## Area chart for decomp
    areaDT <- plotDT$resultCollect$xDecompVec[iterRS == best_iter, !c("mape", "iterRS", "y", "y_pred", "rsq_test")]
    areaDT <- areaDT[, baseline:= rowSums(.SD), .SDcols = c("intercept", set_baseVarName)][, !c("intercept", set_baseVarName), with = F]
    suppressWarnings({areaDT.melt <- melt.data.table(areaDT, id.vars = c("ds"), value.name = "fitted")})
    
    print(ggplot(areaDT.melt, aes(x=ds, y=fitted, fill=variable)) +
            geom_area() +
            labs(title="Sales per predictor over time", 
                 subtitle=paste0("rsq = ", round(best.rsq,4), " , mape = ", round(best.mape,4) ),
                 x="ds",
                 y="Sales")
    )
    
  }
}

#####################################
#### Plotting channel ROI

f.plotChannelROI <- function(plotChannelROI) {
  if (plotChannelROI & exists("best_model")) {
    plotDT <- best_model$resultCollect$xDecompAgg[!is.na(roi), c("rn", "spend", "roi")][order(spend)]
    plotDT[, rn:=as.factor(rn)]
    rn_levels <- plotDT[order(spend)][, rn]
    plotDT[, rn:=factor(rn, levels = rn_levels)]
    
    plotDT <- suppressWarnings(melt.data.table(plotDT, id.vars = "rn"))
    plotDT[, value:=as.double(value)][variable=="spend", value:=round(value, 0)]
    print(ggplot(plotDT, aes(x=reorder(rn, value), y=value, fill = rn)) + 
            geom_bar(stat = "identity", width = 0.5) + 
            scale_fill_brewer(palette = "GnBu") +
            geom_text(aes(label=round(value,2)), hjust=1, size=2.5,fontface = "bold") +
            facet_wrap(~variable, scales = "free") +
            coord_flip() +
            labs(title = "Total historical spend & ROI per channel"
                 ,y="", x="Channels"))
    
  }
}

#####################################
#### Plotting channel ROI

f.plotMAPEConverge <- function(plotMAPEConverge=T) {
  if (plotMAPEConverge) {
    if (exists("model_output")) {
      cumminDT <- model_output$resultCollect$resultHypParam[order(ElapsedAccum)][, mapeAccum:= cummin(mape)]
      cumminDT <- cumminDT[, .(ElapsedAccum= min(ElapsedAccum)), by= mapeAccum]
      cumminDT <- rbind(cumminDT, list(mapeAccum = min(cumminDT$mapeAccum), ElapsedAccum = max(model_output$resultCollect$resultHypParam$ElapsedAccum)))
      cumminDT[, ':='(ElapsedMinute=(ElapsedAccum-shift(ElapsedAccum, type = "lag"))/60)]
      cumminDT[, ':='(mapeDeltePerMinute = abs((mapeAccum-shift(mapeAccum, type = "lag"))/max(mapeAccum)/ElapsedMinute)
                      ,stage = 1:nrow(cumminDT))]
      cumminDT[1, ElapsedMinute := ElapsedAccum/60][1, mapeDeltePerMinute := mapeAccum/ElapsedMinute]
      cumminDT[, max(ElapsedMinute)]/cumminDT[, max(mapeAccum)]
      scale_secY <- cumminDT[, max(ElapsedMinute)]/cumminDT[, max(mapeAccum)]*5
      
      ggplot(data=cumminDT, aes(x = stage)) +
        geom_line(aes(y = mapeAccum) )+
        geom_point(aes(y = mapeAccum)  , color = "red") +
        geom_text(aes(y = mapeAccum, label=round(mapeAccum,2)), position=position_dodge(width=0.9), hjust=-0.5, vjust=0.5, angle=45) +
        
        geom_bar(aes(y = ElapsedMinute /scale_secY) , stat ="identity", position = "dodge") +
        geom_text(aes( y = ElapsedMinute/scale_secY, label=paste0(round(ElapsedMinute,2),"min")), position=position_dodge(width=0.9), hjust=-0.1, vjust=0.5, angle=90) +
        scale_y_continuous(name = "MAPE"
                           ,sec.axis = sec_axis(~.* scale_secY, name = "Elapsed minutes")
                           ,limits=c(0, ceiling(max(cumminDT$mapeAccum))+1)) +
        labs(title="random search (LHS) performance", 
             subtitle=paste0("Total RS iterations:", model_output$resultCollect$iter, "\nTotal elapsed minutes:", round(model_output$resultCollect$elapsed.min,2)),
             x="Minutes spent",
             y="MAPE")      
    }
  }
}

#####################################
#### Plotting hyperparameter sampling

f.plotHyperSamp <- function(plotHyperSamp, channelPlot = NULL) {
  if (plotHyperSamp & exists("model_output")) {
    
    #pairs(lhsOut$initLHS, main = "Pairwise comparison of hyperparameters")
    
    ## get dt for plots
    dtPlotInitLHS <- melt.data.table(model_output$lhsOut$initLHS, measure.vars = local_name.update) # scatter plot
    dtPlotTransLHS <- melt.data.table(model_output$lhsOut$transLHS, measure.vars = local_name.update) # violin plot
    
    ## filter max. 3 channels at one time
    if (is.null(channelPlot)) {
      rowSelect <- str_detect(dtPlotInitLHS$variable, paste0(c(set_mediaVarName[1:3]), collapse = "|"))
    } else {
      if (length(channelPlot)>3) {message("input should have max.3 channels. plotting the first 3 now.")}
      rowSelect <- str_detect(dtPlotInitLHS$variable, paste0(c(channelPlot[1:3]), collapse = "|"))
    }
    dtPlotInitLHS <- dtPlotInitLHS[rowSelect]
    dtPlotTransLHS <- dtPlotTransLHS[rowSelect]
    
    ## plotting
    p1 <- ggplot(dtPlotInitLHS, aes(x = variable, y = value, fill = variable)) +
      geom_jitter(size = 1/log10(set_iter)) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ggtitle("Latin hypercube sampling distribution")
    
    p2 <- ggplot(dtPlotTransLHS, aes(x = variable, y = value, fill = variable)) +
      geom_violinhalf() +
      scale_fill_material_d() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ggtitle("Latin hypercube sampling distribution transformed")
    
    grid.arrange(p1,p2, ncol=1)
    
  }
  #par(mfrow=c(1,1))
}

#####################################
#### Plotting hyperparameter sampling

f.plotBestModDiagnostic <- function(plotBestModDiagnostic) {
  if (plotBestModDiagnostic ==T) {
    
    vecDT <- best_model$resultCollect$xDecompVec
    
    
    p1 <- qplot(x=y_pred, y = y - y_pred, data = vecDT) +
      geom_hline(yintercept = 0) +
      geom_smooth(se = T, method = 'loess', formula = 'y ~ x') + 
      xlab("fitted") + ylab("resid") + ggtitle("fitted vs. residual")
    
    p2 <- ggplot(data = vecDT, aes(sample =  y - y_pred)) +
      xlab("theoretical") + ylab("sample") + ggtitle("QQ") + stat_qq() + stat_qq_line(color = 'blue')
    
    p3 <- qplot(x=y, y = y - y_pred, data = vecDT) +
      geom_hline(yintercept = 0) +
      geom_smooth(se = T, method = 'loess', formula = 'y ~ x') + 
      xlab("observed") + ylab("resid") + ggtitle("observed vs. residual")
    
    grid.arrange(p1,p2,p3, ncol=2)
    
  }
}

#####################################
#### Plotting hyperparameter vs. MAPE

f.plotHypConverge <- function(plotHypConverge, channelPlot = NULL) {
  if (plotHypConverge ==T) {
    
    ## get plot data
    if (exists("best_model")) {
      
      plotDT <- model_output$resultCollect$resultHypParam[, !c("lambdas", "mape", "rsq_test", "pos", "Elapsed", "ElapsedAccum", "iterRS"), with =F]
      best.iter <- model_output$resultCollect$best.iter
      
      best.rsq <- best_model$resultCollect$xDecompAgg$rsq_test
      best.mape <- best_model$resultCollect$xDecompAgg$mape
      
    } else {
      stop("no model_output saved yet. Run at least one epoch to collect result")
    }
    
    ## filter max. 3 channels at one time
    if (is.null(channelPlot)) {
      colSelect <- str_detect(names(plotDT), paste0(c(set_mediaVarName[1:3],"Score"), collapse = "|"))
    } else {
      if (length(channelPlot)>3) {message("input should have max.3 channels. plotting the first 3 now.")}
      #channelPlot <- toupper(channelPlot)
      colSelect <- str_detect(names(plotDT), paste0(c(channelPlot[1:3],"Score"), collapse = "|"))
    }
    plotDT <- plotDT[, colSelect, with =F]
    
    ## trim upper outliers MAPE for better visualisation
    qt_bounds <- quantile(plotDT$Score, probs=c(.25, .75), na.rm = FALSE)
    qt_iqr <- IQR(plotDT$Score)
    #qt_up <-  qt_bounds[2]+1.5*qt_iqr
    qt_low <- qt_bounds[1]-1.5*qt_iqr
    outliers <- plotDT$Score < qt_low
    plotDT <- plotDT[!outliers,]
    
    ## get best mape abline
    plotDT.abline <- melt.data.table(plotDT, id.vars = "Score")
    getAbline <- plotDT.abline[Score == max(Score), value, by = "variable"]
    
    ## plot max. 10k points. Taking random 10k sample when exceeded
    if (nrow(plotDT)>=7000) {
      sampN <- sample(1:nrow(plotDT),7000,replace = F)
      plotDT.melt <- melt.data.table(plotDT[sampN], id.vars = "Score")
    } else {
      plotDT.melt <- melt.data.table(plotDT, id.vars = "Score")
    }
    plotDT.melt[getAbline, abl:= i.value, on = .(variable)]
    
    ## plot
    ggplot(data = plotDT.melt, aes(x=value, y=log(-Score))) +
      geom_point(alpha = 0.1) +
      suppressWarnings(stat_smooth(method = 'gam', formula = y ~ s(x, bs = "cs"), aes(outfit=gamfit<<-..y..))) +
      facet_wrap(~variable, scales = "free") +
      geom_vline(aes(xintercept = abl), colour="coral") +
      labs(title="Hyperparameters vs MAPE", 
           #subtitle=paste0("rsq = ", round(best.rsq,4), " , mape = ", round(best.mape,4) ),
           x="Hyperparameters",
           y="log(MAPE)")
  }
}

#####################################
#### Plotting top 10% model hyperparameter convergence

f.plotHyperBoundOptim <- function(plotHyperBoundOptim, channelPlot = NULL, model_output = model_output, kurt.tuner = 0) {
  
  ## get plot data
  if (exists("model_output") ) {
    
    plotDT <- model_output$resultCollect$resultHypParam[, !c("lambdas", "Score", "rsq_test", "pos", "Elapsed", "ElapsedAccum"), with =F]
    best.iter <- model_output$resultCollect$best.iter
    
    best.rsq <- model_output$resultCollect$xDecompAgg[iterRS == best.iter, rsq_test] 
    best.mape <- model_output$resultCollect$xDecompAgg[iterRS == best.iter, mape] 
    

  
  ## filter max. 3 channels at one time
  if (plotHyperBoundOptim & is.null(channelPlot)) {
    colSelect <- str_detect(names(plotDT), paste0(c(set_mediaVarName[1:3],"mape|iterRS"), collapse = "|"))
  } else if (plotHyperBoundOptim & length(channelPlot)>3){
    message("input should have max.3 channels. plotting the first 3 now.")
    colSelect <- str_detect(names(plotDT), paste0(c(set_mediaVarName[1:3],"mape|iterRS"), collapse = "|"))
  } else if (!plotHyperBoundOptim & is.null(channelPlot)) {
    colSelect <- str_detect(names(plotDT), paste0(c(set_mediaVarName,"mape|iterRS"), collapse = "|"))
  } else {
    colSelect <- str_detect(names(plotDT), paste0(c(channelPlot[1:3],"mape|iterRS"), collapse = "|"))
  }
  plotDT <- plotDT[, colSelect, with =F]
  hyperNamePlot <- names(plotDT[, !c("mape", "iterRS")])
  
  # get top 10% models
  qt_low10 <- quantile(plotDT$mape, probs=c(.1), na.rm = FALSE)
  plotDT <- plotDT[mape < qt_low10]
  
  ## plot max. 10k points. Taking random 10k sample when exceeded
  if (nrow(plotDT)>=10000) {
    sampN <- sample(1:nrow(plotDT),10000,replace = F)
    plotDT.melt <- melt.data.table(plotDT[sampN], id.vars = c("iterRS", "mape"))
  } else {
    plotDT.melt <- melt.data.table(plotDT, id.vars = c("iterRS", "mape"))
  }
  
  ## plot
  plt <- ggplot(data = plotDT.melt) +
    stat_density(geom = "line", aes(x=value), adjust = 1) +
    facet_wrap(~variable, scales = "free") +
    labs(title="Hyperparameter bound optimisation for top 10% MAPE", 
         subtitle=paste0("rsq = ", round(best.rsq,4), " , mape = ", round(best.mape,4) ),
         x="Hyperparameters",
         y="Density") # print(plt)
  
  plt.data <- ggplot_build(plt)$data[[1]]
  nVar <- length(levels(plotDT.melt$variable))
  denLen <- length(plt.data$y)/nVar
  
  denYList <- lapply(1:nVar,  FUN = function(x) plt.data$y[(x-1)*denLen + (1:denLen)])
  names(denYList) <- hyperNamePlot
  denXList <- lapply(1:nVar,  FUN = function(x) plt.data$x[(x-1)*denLen + (1:denLen)])
  names(denXList) <- hyperNamePlot
  
  kurt <- sapply(denYList, kurtosis, method = "fisher")
  kurt <- kurt + 1.6 + 0.2 * kurt.tuner# adjust kurt level
  optim.found <- kurt > 0
  if (activate_hyperBoundLocalTuning==T) {
    bounds_whichfixed_plot <- bounds_whichfixed[hyperNamePlot]
    optim.found <- (optim.found * !bounds_whichfixed_plot) > 0
  } else {
    bounds_whichfixed_plot <- rep(FALSE, nVar)
  }
  boundRecomm <- t(mapply(function(x,y) {
    xMode <- x[which.max(y)]
    c(mode = xMode,
      low = ifelse((xMode-sd(x))< min(x), min(x), xMode-sd(x)),
      up = ifelse(xMode+sd(x) > max(x), max(x), xMode+sd(x))
    )
  }, x=denXList, y=denYList
  ))
  
  nLen <- nrow(plotDT.melt)/nVar
  plotDT.melt[, ':='(variable= factor(plotDT.melt$variable
                                      ,levels = levels(plotDT.melt$variable)
                                      ,labels = paste0(levels(plotDT.melt$variable) ,"\nkurtosis adj.: ",round(kurt,4)
                                                       ,"\nlocal optim: ", optim.found))
                     ,mode = rep(boundRecomm[,1], each = nLen)
                     ,low = rep(boundRecomm[,2], each = nLen)
                     ,up = rep(boundRecomm[,3], each = nLen)
  )]
  
  plt.out <- data.table(variable = levels(plotDT.melt$variable)
                        ,kurt = kurt
                        ,optim.found = optim.found
                        ,mode = boundRecomm[,1]
                        ,low = boundRecomm[,2]
                        ,up = boundRecomm[,3])
  
  y_exclude <- as.vector(mapply(function(y_start,y_end) {
    return(y_start:y_end)
  }, y_start=((which(bounds_whichfixed_plot)-1)*denLen+1), y_end = (which(bounds_whichfixed_plot)*denLen)))
  y_max <- max(plt.data$y[setdiff(1:nrow(plt.data), y_exclude)])
  
  if (plotHyperBoundOptim ==T) {
    print(plt + scale_y_continuous(limits = c(0, ceiling(y_max))) + 
            geom_vline(data= plt.out, mapping = aes(xintercept = mode), colour="blue") + 
            geom_vline(data= plt.out, mapping = aes(xintercept = low), colour="blue", linetype="dotted") +
            geom_vline(data= plt.out, mapping = aes(xintercept = up), colour="blue", linetype="dotted") +
            geom_text(data = plt.out, aes(x = mode, y = y_max, angle = 90, vjust = -0.5, hjust= 1, label = paste0("mode: ",round(mode,4))), colour="blue") +
            geom_text(data = plt.out, aes(x = low, y = y_max, angle = 90, vjust = -0.5, hjust= 1, label = paste0("low: ",round(low,4))), colour="blue") +
            geom_text(data = plt.out, aes(x = up, y = y_max, angle = 90, vjust = -0.5, hjust= 1, label = paste0("up: ",round(up,4))), colour="blue"))
  }
  
  plt.out[, variable := str_sub(plt.out$variable, rep(1,nVar), str_locate(plt.out$variable, "\n")[,1]-1)]
  return(plt.out)
  
  } else {
    warning("run f.mmmRobyn first to get model_output for this plot")
  }
}

#####################################
#### Plotting spend-reach model fit

f.plotSpendModel <- function(plotSpendModel) {
  if(any(costSelector) & plotSpendModel) {
    grid.arrange(grobs = plotNLSCollect
                 ,ncol= ifelse(length(plotNLSCollect)<=3, length(plotNLSCollect), 3)
                 ,top = "Spend-reach fitting with Michaelis-Menten model")
  } else if (!any(costSelector) & plotSpendModel) {
    message("no spend model needed. all media variables used for mmm are spend variables ")
  }
}

#####################################
#### Plotting optimiser result comparison

f.plotOptimiser <- function(plotOptimiser) {
  if (plotOptimiser & exists("optim_result")) {
    
    plotDT_total <- optim_result$dt_optimOut
    
    # ROI comparison plot
    
    plotDT_roi <- plotDT_total[, c("channels", "initRoiUnit", "optmRoiUnit")][order(initRoiUnit)]
    plotDT_roi[, channels:=as.factor(channels)]
    chn_levels <- plotDT_roi[, as.character(channels)]
    plotDT_roi[, channels:=factor(channels, levels = chn_levels)]
    setnames(plotDT_roi, names(plotDT_roi), new = c("channel", "initial roi", "optimised roi"))
    
    plotDT_roi <- suppressWarnings(melt.data.table(plotDT_roi, id.vars = "channel", value.name = "roi"))
    print(ggplot(plotDT_roi, aes(x=channel, y=roi, fill = channel)) +
            geom_bar(stat = "identity", width = 0.5) +
            coord_flip() +
            facet_wrap(~variable, scales = "free") +
            scale_fill_brewer(palette = "GnBu") +
            geom_text(aes(label=round(roi,2), hjust=1, fontface = "bold")) +
            labs(title = "Optimised media mix ROI comparison"
                 ,subtitle = paste0("Total spend increases ", plotDT_total[, round(mean(optmSpendUnitTotalDelta)*100,1)], "%"
                                    ,"\nTotal response increases ", plotDT_total[, round(mean(optmResponseUnitTotalLift)*100,1)], "% with optimised spend allocation"
                 )
                 ,y="", x="Channels"))
    
    # Response comparison plot
    plotDT_resp <- plotDT_total[, c("channels", "initResponseUnit", "optmResponseUnit")][order(initResponseUnit)]
    plotDT_resp[, channels:=as.factor(channels)]
    chn_levels <- plotDT_resp[, as.character(channels)]
    plotDT_resp[, channels:=factor(channels, levels = chn_levels)]
    setnames(plotDT_resp, names(plotDT_resp), new = c("channel", "initial response / time unit", "optimised response / time unit"))
    
    plotDT_resp <- suppressWarnings(melt.data.table(plotDT_resp, id.vars = "channel", value.name = "response"))
    print(ggplot(plotDT_resp, aes(x=channel, y=response, fill = channel)) +
            geom_bar(stat = "identity", width = 0.5) +
            coord_flip() +
            facet_wrap(~variable, scales = "free") +
            scale_fill_brewer(palette = "GnBu") +
            geom_text(aes(label=round(response,0), hjust=1, fontface = "bold")) +
            labs(title = "Optimised media mix response comparison"
                 ,subtitle = paste0("Total spend increases ", plotDT_total[, round(mean(optmSpendUnitTotalDelta)*100,1)], "%"
                                    ,"\nTotal response increases ", plotDT_total[, round(mean(optmResponseUnitTotalLift)*100,1)], "% with optimised spend allocation"
                 )
                 ,y="", x="Channels"))
    
    # budget share comparison plot
    plotDT_share <- plotDT_total[, c("channels", "initSpendShare", "optmSpendShareUnit")][order(initSpendShare)]
    plotDT_share[, channels:=as.factor(channels)]
    chn_levels <- plotDT_share[, as.character(channels)]
    plotDT_share[, channels:=factor(channels, levels = chn_levels)]
    setnames(plotDT_share, names(plotDT_share), new = c("channel", "initial spend share", "optimised spend share"))
    
    plotDT_share <- suppressWarnings(melt.data.table(plotDT_share, id.vars = "channel", value.name = "spend_share"))
    print(ggplot(plotDT_share, aes(x=channel, y=spend_share, fill = channel)) +
            geom_bar(stat = "identity", width = 0.5) +
            coord_flip() +
            facet_wrap(~variable, scales = "free") +
            scale_fill_brewer(palette = "GnBu") +
            geom_text(aes(label=paste0(round(spend_share*100,2),"%")), hjust=1, size=2.5,fontface = "bold") +
            labs(title = "Optimised media mix budget allocation"
                 ,subtitle = paste0("Total spend increases ", plotDT_total[, round(mean(optmSpendUnitTotalDelta)*100,1)], "%"
                                    ,"\nTotal response increases ", plotDT_total[, round(mean(optmResponseUnitTotalLift)*100,1)], "% with optimised spend allocation"
                 )
                 ,y="", x="Channels"))
    
  }
}


