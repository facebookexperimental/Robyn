# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################
###### Data transformation and helper functions
########################################################################

################################################################
#### Define basic condition check function

f.checkConditions <- function(dt_transform) {
  
  if (activate_prophet & !all(set_prophet %in% c("trend","season", "weekday", "holiday"))) {
    stop("set_prophet must be 'trend', 'season', 'weekday' or 'holiday")
  }
  
  if (activate_baseline) {
    if(length(set_baseVarName) != length(set_baseVarSign)) {stop("set_baseVarName and set_baseVarSign have to be the same length")}
  }
  
  if (!exists("set_mediaVarName")) {
    stop("set_mediaVarName must be specified")
  } else {
    if(length(set_mediaVarName) != length(set_mediaVarSign)) {stop("set_mediaVarName and set_mediaVarSign have to be the same length")}
    if(!all(c(set_prophetVarSign, set_baseVarSign, set_mediaVarSign) %in% c("positive", "negative", "default"))) {
      stop("set_prophetVarSign, set_baseVarSign & set_mediaVarSign must be 'positive', 'negative' or 'default'")}
  }
  
  if(activate_calibration) {
    if(nrow(set_lift)==0 | !exists("set_lift")) {
      stop("please provide lift result or set activate_calibration = FALSE")
    }
    if ((min(set_lift$liftStartDate) < min(dt_transform$ds)) | (max(set_lift$liftEndDate) >  (max(dt_transform$ds) + dayInterval-1))) {
      stop("we recommend you to only use lift results conducted within your MMM input data date range")
    }
    if (set_iter < 500 | set_trial < 30) {message("you are calibrating MMM. we recommend to run at least 500 iterations per trial and at least 60 trials at the beginning")}
  } else {
    if (set_iter < 500 | set_trial < 30) {message("we recommend to run at least 500 iterations per trial and at least 30 trials at the beginning")}
  }
  
  if((adstock %in% c("geometric", "weibull")) == F) {stop("adstock must be 'geometric' or 'weibull'")}
  
  num_hp_channel <- ifelse(adstock == "geometric", 3, 4)
  if( all(str_detect(names(set_hyperBoundLocal) ,paste0(local_name, collapse = "|")))==F & length(unique(names(set_hyperBoundLocal))) != length(set_mediaVarName)*num_hp_channel) {
    local_names <- f.getHyperNames()
    stop("set_hyperBoundLocal has incorrect hyperparameters. names of hyperparameters must be: \n", paste(local_names, collapse = ", "))
  }
  
  if(any(apply(dt_transform, 2, function(x) any(is.na(x) | is.infinite(x))))) {stop("input datafrom dt has NA or Inf")}
  
}

################################################################
#### Define major input data transformation function

f.inputWrangling <- function(dt_transform = dt_input) {
  
  ## check date format
  tryCatch({
    dateCheck <- as.matrix(dt_transform[,  set_dateVarName, with=F])
    dateCheck <- as.Date(dateCheck)
  },
  error= function() {
    stop("input date variable should have format '2020-01-01'")
  })
  
  ## check variables existence
  
  if (!activate_prophet) {
    assign("set_prophet", NULL, envir = .GlobalEnv)
    assign("set_prophetVarSign", NULL, envir = .GlobalEnv)
  }
  
  if (!activate_baseline) {
    assign("set_baseVarName", NULL, envir = .GlobalEnv)
    assign("set_baseVarSign", NULL, envir = .GlobalEnv)
  }
  
  
  if (!activate_calibration) {
    assign("set_lift", NULL, envir = .GlobalEnv)
  }
  
  if (!exists("set_mediaSpendName")) {stop("set_mediaSpendName must be specified")
  } else if(length(set_mediaVarName) != length(set_mediaSpendName)) {
    stop("set_mediaSpendName and set_mediaVarName have to be the same length and same order")}
  
  
  setnames(dt_transform, set_dateVarName, "ds")
  dt_transform[, ':='(ds= as.Date(ds))]
  
  setnames(dt_transform, set_depVarName, "depVar") #; set_depVarName <- "depVar"
  #indepName <- c(set_prophet, set_baseVarName, set_mediaVarName)
  
  #hypName <- c("thetas", "shapes", "scales", "alphas", "gammas", "lambdas") # defind hyperparameter names
  dayInterval <- as.integer(difftime(sort(unique(dt_transform$ds))[2], sort(unique(dt_transform$ds))[1], units = "days"))
  intervalType <- if(dayInterval==1) {"day"} else if (dayInterval==7) {"week"} else if (dayInterval %in% 28:31) {"month"} else {stop("input data has to be daily, weekly or monthly")}
  assign("dayInterval", dayInterval, envir = .GlobalEnv)
  mediaVarCount <- length(set_mediaVarName)
  
  ################################################################
  #### model reach metric from spend
  
  mediaCostFactor <- unlist(dt_input[, lapply(.SD, sum), .SDcols = set_mediaSpendName] / dt_input[, lapply(.SD, sum), .SDcols = set_mediaVarName])
  names(mediaCostFactor) <- set_mediaVarName
  costSelector <- !(set_mediaSpendName == set_mediaVarName)
  
  if (any(costSelector)) {
    modNLSCollect <- list()
    yhatCollect <- list()
    plotNLSCollect <- list()
    for (i in 1:mediaVarCount) {
      if (costSelector[i]) {
        dt_spendModInput <- dt_input[, c(set_mediaSpendName[i],set_mediaVarName[i]), with =F]
        setnames(dt_spendModInput, names(dt_spendModInput), c("spend", "reach"))
        #dt_spendModInput <- dt_spendModInput[spend !=0 & reach != 0]
        
        # scale 0 spend and reach to a tiny number
        dt_spendModInput[, spend:=as.numeric(spend)][spend==0, spend:=0.01] # remove spend == 0 to avoid DIV/0 error
        dt_spendModInput[, reach:=as.numeric(reach)][reach==0, reach:=spend / mediaCostFactor[i]] # adapt reach with avg when spend == 0
        
        # mod_nls <- nls(reach ~ SSmicmen(spend, Vmax, Km)
        #                ,data = dt_spendModInput
        #                ,control = nls.control(minFactor=1/2048, warnOnly = T))
        
        # estimate starting values for nls
        # modLM <- lm(log(reach) ~ spend, dt_spendModInput)
        # nlsStartVal <- list(Vmax = exp(coef(modLM)[1]), Km = coef(modLM)[2])
        # nlsStartVal <- list(Vmax = dt_spendModInput[, max(reach)/2], Km = dt_spendModInput[, max(reach)])
        # run nls model
        # modNLS <- nlsLM(reach ~ Vmax * spend/(Km + spend), #Michaelis-Menten model Vmax * spend/(Km + spend)
        #                data = dt_spendModInput,
        #                start = nlsStartVal
        #                ,control = nls.control(warnOnly = T)
        # )
        
        modNLS <- tryCatch(
          {
            nlsStartVal <- list(Vmax = dt_spendModInput[, max(reach)/2], Km = dt_spendModInput[, max(reach)])
            modNLS <- minpack.lmnlsLM(reach ~ Vmax * spend/(Km + spend), #Michaelis-Menten model Vmax * spend/(Km + spend)
                                      data = dt_spendModInput,
                                      start = nlsStartVal
                                      ,control = nls.control(warnOnly = T))
          },
          error=function(cond) {
            nlsStartVal <- list(Vmax=1, Km=1)
            suppressWarnings(modNLS <- nlsLM(reach ~ Vmax * spend/(Km + spend), #Michaelis-Menten model Vmax * spend/(Km + spend)
                                             data = dt_spendModInput,
                                             start = nlsStartVal
                                             ,control = nls.control(warnOnly = T)))
            #warning("default start value for nls out of range. using c(1,1) instead")
            return(modNLS)
          }
        )
        
        yhatNLS <- predict(modNLS)
        modNLSSum <- summary(modNLS)
        
        # QA nls model prediction
        yhatNLSQA <- modNLSSum$coefficients[1,1] * dt_spendModInput$spend / (modNLSSum$coefficients[2,1] + dt_spendModInput$spend)
        identical(yhatNLS, yhatNLSQA)
        
        # build lm comparison model
        modLM <- lm(reach ~ spend-1, data = dt_spendModInput)
        yhatLM <- predict(modLM)
        modLMSum <- summary(modLM)
        
        # compare NLS & LM, takes LM if NLS fits worse
        rsq_nls <- f.rsq(dt_spendModInput$reach, yhatNLS)
        rsq_lm <- f.rsq(dt_spendModInput$reach, yhatLM) #reach = v  * spend / (k + spend)
        costSelector[i] <- rsq_nls > rsq_lm
        
        modNLSCollect[[set_mediaVarName[i]]] <- data.table(channel = set_mediaVarName[i],
                                                           Vmax = modNLSSum$coefficients[1,1],
                                                           Km = modNLSSum$coefficients[2,1],
                                                           aic_nls = AIC(modNLS),
                                                           aic_lm = AIC(modLM),
                                                           bic_nls = BIC(modNLS),
                                                           bic_lm = BIC(modLM),
                                                           rsq_nls = rsq_nls,
                                                           rsq_lm = rsq_lm,
                                                           coef_lm = coef(modLMSum)[1]
        )
        
        dt_plotNLS <- data.table(channel = set_mediaVarName[i],
                                 yhatNLS = if(costSelector[i]) {yhatNLS} else {yhatLM},
                                 yhatLM = yhatLM,
                                 y = dt_spendModInput$reach,
                                 x = dt_spendModInput$spend)
        dt_plotNLS <- melt.data.table(dt_plotNLS, id.vars = c("channel", "y", "x"), variable.name = "models", value.name = "yhat")
        dt_plotNLS[, models:= str_remove(tolower(models), "yhat")]
        
        yhatCollect[[set_mediaVarName[i]]] <- dt_plotNLS
        
        # create plot
        plotNLSCollect[[set_mediaVarName[i]]] <- ggplot(dt_plotNLS, aes(x=x, y=y, color = models)) +
          geom_point() +
          geom_line(aes(y=yhat, x=x, color = models)) +
          labs(subtitle = paste0("y=",set_mediaVarName[i],", x=", set_mediaSpendName[i],
                                 "\nnls: aic=", round(AIC(if(costSelector[i]) {modNLS} else {modLM}),0), ", rsq=", round(if(costSelector[i]) {rsq_nls} else {rsq_lm},4),
                                 "\nlm: aic= ", round(AIC(modLM),0), ", rsq=", round(rsq_lm,4)),
               x = "spend",
               y = "reach"
          ) +
          theme(legend.position = 'bottom')
        
      }
    }
    
    modNLSCollect <- rbindlist(modNLSCollect)
    yhatNLSCollect <- rbindlist(yhatCollect)
    yhatNLSCollect[, ds:= rep(dt_transform$ds, nrow(yhatNLSCollect)/nrow(dt_transform))]
    assign("plotNLSCollect", plotNLSCollect, envir = .GlobalEnv)
    assign("modNLSCollect", modNLSCollect, envir = .GlobalEnv)
    assign("yhatNLSCollect", yhatNLSCollect, envir = .GlobalEnv)
    
  }
  
  getSpendSum <- dt_input[, lapply(.SD, sum), .SDcols=set_mediaSpendName]
  names(getSpendSum) <- set_mediaVarName
  getSpendSum <- suppressWarnings(melt.data.table(getSpendSum, measure.vars= set_mediaVarName, variable.name = "rn", value.name = "spend"))
  
  assign("mediaCostFactor", mediaCostFactor, envir = .GlobalEnv)
  assign("costSelector", costSelector, envir = .GlobalEnv)
  assign("getSpendSum", getSpendSum, envir = .GlobalEnv)
  
  
  ################################################################
  #### clean & aggregate data
  
  all_name <- unique(c("ds", "depVar", set_prophet, set_baseVarName, set_mediaVarName #, set_keywordsVarName, set_mediaSpendName
  ))
  all_mod_name <- c("ds", "depVar", set_prophet, set_baseVarName, set_mediaVarName)
  if(!identical(all_name, all_mod_name)) {stop("Input variables must have unique names")}
  
  ## transform all factor variables
  if (exists("set_factorVarName")) {
    if (length(set_factorVarName)>0) {
      #set_factorVarName <- toupper(set_factorVarName)
      dt_transform[, (set_factorVarName):= as.factor(get(set_factorVarName)) ]
    }
  } else {
    assign("set_factorVarName", NULL, envir = .GlobalEnv)
  }
  
  ################################################################
  #### Obtain prophet trend, seasonality and changepoints
  
  if (activate_prophet) {
    
    if(length(set_prophet) != length(set_prophetVarSign)) {stop("set_prophet and set_prophetVarSign have to be the same length")}
    if(any(length(set_prophet)==0, length(set_prophetVarSign)==0)) {stop("if activate_prophet == TRUE, set_prophet and set_prophetVarSign must to specified")}
    if(!(set_country %in% dt_holidays$country)) {stop("set_country must be already included in the holidays.csv and as ISO 3166-1 alpha-2 abbreviation")}
    
    recurrance <- dt_transform[, .(ds = ds, y = depVar)]
    use_trend <- any(str_detect("trend", set_prophet))
    use_season <- any(str_detect("season", set_prophet))
    use_weekday <- any(str_detect("weekday", set_prophet))
    use_holiday <- any(str_detect("holiday", set_prophet))
    
    if (intervalType == "day") {
      
      holidays <- dt_holidays
      
    } else if (intervalType == "week") {
      
      weekStartInput <- weekdays(dt_transform[1, ds])
      weekStartMonday <- if(weekStartInput=="Monday") {TRUE} else if (weekStartInput=="Sunday") {FALSE} else {stop("week start has to be Monday or Sunday")}
      dt_holidays[, dsWeekStart:= cut(as.Date(ds), breaks = intervalType, start.on.monday = weekStartMonday)]
      holidays <- dt_holidays[, .(ds=dsWeekStart, holiday, country, year)]
      holidays <- holidays[, lapply(.SD, paste0, collapse="#"), by = c("ds", "country", "year"), .SDcols = "holiday"]
      
    } else if (intervalType == "month") {
      
      monthStartInput <- all(day(dt_transform[, ds]) ==1)
      if (monthStartInput==FALSE) {stop("monthly data should have first day of month as datestampe, e.g.'2020-01-01' ")}
      dt_holidays[, dsMonthStart:= cut(as.Date(ds), intervalType)]
      holidays <- dt_holidays[, .(ds=dsMonthStart, holiday, country, year)]
      holidays <- holidays[, lapply(.SD, paste0, collapse="#"), by = c("ds", "country", "year"), .SDcols = "holiday"]
      
    }
    
    modelRecurrance<-prophet(recurrance
                             ,holidays = if(use_holiday) {holidays[country==set_country]} else {NULL}
                             ,yearly.seasonality = use_season
                             ,weekly.seasonality = use_weekday
                             ,daily.seasonality= F
                             #,changepoint.range = 0.8
                             #,seasonality.mode = 'multiplicative'
                             #,changepoint.prior.scale = 0.1
    )
    
    futureDS <- make_future_dataframe(modelRecurrance, periods=1, freq = intervalType)
    forecastRecurrance <- predict(modelRecurrance, futureDS)
    assign("modelRecurrance", modelRecurrance, envir = .GlobalEnv)
    assign("forecastRecurrance", forecastRecurrance, envir = .GlobalEnv)
    #plot(modelRecurrance, forecastRecurrance)
    
    if (use_trend) {
      fc_trend <- forecastRecurrance$trend[1:NROW(recurrance)]
      recurrance[, trend := scale(fc_trend, center = min(fc_trend), scale = F) + 1]
      dt_transform[, trend := recurrance$trend]
    }
    if (use_season) {
      fc_season <- forecastRecurrance$yearly[1:NROW(recurrance)]
      recurrance[, seasonal := scale(fc_season, center = min(fc_season), scale = F) + 1]
      dt_transform[, season := recurrance$seasonal]
    }
    if (use_weekday) {
      fc_weekday <- forecastRecurrance$weekly[1:NROW(recurrance)]
      recurrance[, weekday := scale(fc_weekday, center = min(fc_weekday), scale = F) + 1]
      dt_transform[, weekday := recurrance$weekday]
    }
    if (use_holiday) {
      fc_holiday <- forecastRecurrance$holidays[1:NROW(recurrance)]
      recurrance[, holidays := scale(fc_holiday, center = min(fc_holiday), scale = F) + 1]
      dt_transform[, holiday := recurrance$holidays]
    }
  }
  
  ################################################################
  #### Finalize input
  
  #dt <- dt[, all_name, with = F]
  dt_transform <- dt_transform[, all_mod_name, with = F]
  
  f.checkConditions(dt_transform)
  
  return(dt_transform)
}

################################################################
#### Define hyperparameter names extraction function

f.getHyperNames <- function() {
  global_name <- c("thetas",  "shapes",  "scales",  "alphas",  "gammas",  "lambdas")
  if (adstock == "geometric") {
    local_name <- sort(apply(expand.grid(set_mediaVarName, global_name[global_name %like% 'thetas|alphas|gammas']), 1, paste, collapse="_"))
  } else if (adstock == "weibull") {
    local_name <- sort(apply(expand.grid(set_mediaVarName, global_name[global_name %like% 'shapes|scales|alphas|gammas']), 1, paste, collapse="_"))
  }
  return(local_name)
}

################################################
#### Define adstock geometric function

f.adstockGeometric <- function(x, theta) {
  x_decayed <- c(x[1] ,rep(0, length(x)-1))
  for (xi in 2:length(x_decayed)) {
    x_decayed[xi] <- x[xi] + theta * x_decayed[xi-1]
  }
  return(x_decayed)
}

################################################
#### Define adstock weibull function

f.adstockWeibull <- function(x, shape , scale) {
  x.n <- length(x)
  x_bin <- 1:x.n
  scaleTrans = round(quantile(x_bin, scale),0) #
  thetaVec <- c(1, 1-pweibull(head(x_bin, -1), shape = shape, scale = scaleTrans)) # plot(thetaVec)
  thetaVecCum <- cumprod(thetaVec)  # plot(thetaVecCum)
  
  x_decayed <- mapply(function(x, y) {
    x.vec <- c(rep(0,y-1), rep(x, x.n-y+1))
    thetaVecCumLag <- shift(thetaVecCum, y-1, fill = 0)
    x.matrix <- cbind(x.vec, thetaVecCumLag) #  plot(x.vec)
    x.prod <- apply(x.matrix, 1, prod)
    return(x.prod)
  }, x=x , y=x_bin)
  x_decayed <- rowSums(x_decayed)
  
  return(list(x_decayed=x_decayed, thetaVecCum = thetaVecCum))
}

################################################
#### Define transformation function

f.transformation <- function (x, theta= NULL, shape= NULL, scale= NULL, alpha=NULL, gamma=NULL, alternative = adstock, stage = 3) {
  
  ## step 1: add decay rate
  
  if (alternative == "geometric") {
    x_decayed <- f.adstockGeometric(x, theta)
    
    if (stage == "thetaVecCum") {
      thetaVecCum <- theta
      for (t in 2:length(x)) {thetaVecCum[t] <- thetaVecCum[t-1]*theta} # plot(thetaVecCum)
    }
    
  } else if (alternative == "weibull") {
    x_list <- f.adstockWeibull(x, shape, scale)
    x_decayed <- x_list$x_decayed # plot(x_decayed)
    
    if (stage == "thetaVecCum") {
      thetaVecCum <- x_list$thetaVecCum # plot(thetaVecCum)
    }
    
  } else {
    print("alternative must be geometric or weibull")
  }
  
  ## step 2: normalize decayed independent variable # deprecated
  #x_normalized <- scale(x_decayed, center =F) # plot(x_normalized) summary(x_normalized)
  x_normalized <- x_decayed
  
  ## step 3: s-curve transformation
  gammaTrans <- round(quantile(seq(range(x_normalized)[1], range(x_normalized)[2], length.out = 100), gamma),4)
  x_scurve <-  x_normalized**alpha / (x_normalized**alpha + gammaTrans**alpha) # plot(x_scurve) summary(x_scurve)
  
  if (stage == 1) {
    x_out <- x_decayed
  } else if (stage == 2) {
    x_out <- x_normalized
  } else if (stage ==3) {
    x_out <- x_scurve
  } else if (stage == "thetaVecCum") {
    x_out <- thetaVecCum
  } else {stop("stage must be 1, 2 or 3, indicating adstock, normalization & scurve stages of transformation") }
  
  if (sum(is.nan(x_out))>0) {stop("hyperparameters out of range. theta range: 0-1 (excl.1), shape range: 0-5 (excl.0), alpha range: 0-5 (excl.0),  gamma range: 0-1 (excl.0)")}
  return(x_out)
}


################################################
#### Define r-squared function

f.rsq <- function(true, predicted) {
  sse <- sum((predicted - true)^2)
  sst <- sum((true - mean(true))^2)
  rsq <- 1 - sse / sst
  return(rsq)
}

################################################
#### Define ridge lambda sequence function

f.lambdaRidge <- function(x, y, seq_len = 100, lambda_min_ratio = 0.0001) {
  mysd <- function(y) sqrt(sum((y-mean(y))^2)/length(y))
  sx <- scale(x,scale=apply(x, 2, mysd))
  sx <- as.matrix(sx, ncol=ncol(x), nrow=nrow(x))
  #sy <- as.vector(scale(y, scale=mysd(y)))
  sy <- y
  lambda_max <- max(abs(colSums(sx*sy))) / (0.001 * nrow(x)) # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)
  
  lambda_max_log <- log(lambda_max)
  log_step <- (log(lambda_max)-log(lambda_max * lambda_min_ratio)) / (seq_len-1)
  log_seq <- seq(log(lambda_max) , log(lambda_max*lambda_min_ratio), length.out = seq_len)
  lambda_seq <- exp(log_seq)
  return(lambda_seq)
}

################################################
#### Define model decomposition function

f.decomp <- function(coefs, dt_modAdstocked, x, y_pred, i) {
  
  ## input for decomp
  y <- dt_modAdstocked$depVar
  indepVar <- dt_modAdstocked[, (setdiff(names(dt_modAdstocked), "depVar")), with = F]
  x <- as.data.table(x)
  intercept <- coefs[1]
  indepVarName <- names(indepVar)
  indepVarCat <- indepVarName[sapply(indepVar, is.factor)]
  
  ## decomp x
  xDecomp <- data.table(mapply(function(regressor, coeff) {regressor*coeff}, regressor = x, coeff= coefs[-1]))
  xDecomp <- cbind(data.table(intercept = rep(intercept, nrow(xDecomp))), xDecomp)
  #xDecompOut <- data.table(sapply(indepVarName, function(x) xDecomp[, rowSums(.SD,), .SDcols = str_which(names(xDecomp), x)]))
  xDecompOut <- cbind(data.table(ds = dt_mod$ds, y = y, y_pred = y_pred) ,xDecomp)
  
  ## QA decomp
  y_hat <- rowSums(xDecomp)
  errorTerm <- y_hat - y_pred
  if (prod(round(y_pred) == round(y_hat)) == 0) {cat("\n### attention for loop", i,": manual decomp is not matching linear model prediction. Deviation is", mean(errorTerm / y)*100,"% ### \n")}
  
  ## output decomp
  y_hat.scaled <- rowSums(abs(xDecomp))
  xDecompOutPerc.scaled <- abs(xDecomp)/y_hat.scaled
  xDecompOut.scaled <- y_hat*xDecompOutPerc.scaled
  
  xDecompOutAgg <- sapply(xDecompOut[, c("intercept", indepVarName), with =F], function(x) sum(x))
  xDecompOutAggPerc <- xDecompOutAgg / sum(y_hat)
  xDecompOutAggPerc.scaled <- abs(xDecompOutAggPerc)/sum(abs(xDecompOutAggPerc))
  xDecompOutAgg.scaled <- sum(xDecompOutAgg)*xDecompOutAggPerc.scaled
  
  coefsOut <- data.table(coefs, keep.rownames = T)
  coefsOut[, rn := if (length(indepVarCat) == 0) {rn} else {sapply(indepVarCat, function(x) str_replace(coefsOut$rn, paste0(x,".*"), x))}]
  coefsOut <- coefsOut[, .(coef = mean(s0)), by = rn]
  
  decompOutAgg <- cbind(coefsOut, data.table(xDecompAgg = xDecompOutAgg
                                             ,xDecompPerc = xDecompOutAggPerc
                                             ,xDecompAgg.scaled = xDecompOutAgg.scaled
                                             ,xDecompPerc.scaled = xDecompOutAggPerc.scaled))
  decompOutAgg[, pos:= xDecompAgg>=0]
  
  decompCollect <- list(xDecompVec= xDecompOut, xDecompVec.scaled=xDecompOut.scaled, xDecompAgg = decompOutAgg)
  
  return(decompCollect)
} ## decomp end

################################################
#### Define lift calibration function

f.calibrateLift <- function(decompCollect, set_lift) {
  
  check_set_lift <- any(sapply(set_lift$channel, function(x) any(str_detect(x, set_mediaVarName)))==F) #check if any lift channel doesnt have media var
  if (check_set_lift) {stop("set_lift channels must have media variable")}
  ## prep lift input
  getLiftMedia <- unique(set_lift$channel)
  getDecompVec <- decompCollect$xDecompVec
  
  ## loop all lift input
  liftCollect <- list()
  for (m in 1:length(getLiftMedia)) { # loop per lift channel
    
    liftWhich <- str_which(set_lift$channel, getLiftMedia[m])
    
    liftCollect2 <- list()
    for (lw in 1:length(liftWhich)) { # loop per lift test per channel
      
      ## get lift period subset
      liftStart <- set_lift[liftWhich[lw], liftStartDate]
      liftEnd <- set_lift[liftWhich[lw], liftEndDate]
      liftPeriodVec <- getDecompVec[ds >= liftStart & ds <= liftEnd, c("ds", getLiftMedia[m]), with = F]
      liftPeriodVecDependent <- getDecompVec[ds >= liftStart & ds <= liftEnd, c("ds", "y"), with = F]
      
      ## scale decomp
      mmmDays <- nrow(liftPeriodVec) * 7
      liftDays <- as.integer(liftEnd- liftStart + 1)
      y_hatLift <- sum(unlist(getDecompVec[, -1])) # total pred sales
      x_decompLift <- sum(liftPeriodVec[,2])
      x_decompLiftScaled <- x_decompLift / mmmDays * liftDays
      y_scaledLift <- liftPeriodVecDependent[, sum(y)] / mmmDays * liftDays
      
      ## output
      liftCollect2[[lw]] <- data.table(liftMedia = getLiftMedia[m] ,
                                       liftStart = liftStart,
                                       liftEnd = liftEnd,
                                       liftAbs = set_lift[liftWhich[lw], liftAbs],
                                       decompAbsScaled = x_decompLiftScaled,
                                       dependent = y_scaledLift)
    }
    liftCollect[[m]] <- rbindlist(liftCollect2)
  }
  
  ## get mape_lift
  liftCollect <- rbindlist(liftCollect)[, mape_lift := abs((decompAbsScaled - liftAbs) / liftAbs)]
  return(liftCollect)
}


########################################################################
###### Major MMM function
########################################################################


#####################################
#### Define refit function
f.refit <- function(x_train, y_train, x_test, y_test, lambda, lower.limits, upper.limits) {
  mod <- glmnet(
    x_train
    ,y_train
    ,family = "gaussian"
    ,alpha = 0 #0 for ridge regression
    ,lambda = lambda # https://stats.stackexchange.com/questions/138569/why-is-lambda-within-one-standard-error-from-the-minimum-is-a-recommended-valu
    ,lower.limits = lower.limits
    ,upper.limits = upper.limits
  ) # coef(mod)
  
  ## drop intercept if negative
  if (coef(mod)[1] <0) {
    mod <- glmnet(
      x_train
      ,y_train
      ,family = "gaussian"
      ,alpha = 0 #0 for ridge regression
      ,lambda = lambda
      ,lower.limits = lower.limits
      ,upper.limits = upper.limits
      ,intercept = FALSE
    ) # coef(mod)
  } #; plot(mod); print(mod)
  
  y_trainPred <- predict(mod, s = lambda, newx = x_train)
  rsq_train<- f.rsq(true = y_train, predicted = y_trainPred); rsq_train
  
  y_testPred <- predict(mod, s = lambda, newx = x_test)
  rsq_test <- f.rsq(true = y_test, predicted = y_testPred); rsq_test
  
  mape_mod<- mean(abs((y_test - y_testPred)/y_test)* 100); mape_mod
  coefs <- as.matrix(coef(mod))
  y_pred <- c(y_trainPred, y_testPred)
  
  nrmse_train <- sqrt(mean(sum((y_train - y_trainPred)^2))) / (max(y_train) - min(y_train)) # mean(y_train) sd(y_train)
  nrmse_test <- sqrt(mean(sum((y_test - y_testPred)^2))) / (max(y_test) - min(y_test)) # mean(y_test) sd(y_test)
  
  mod_out <- list(rsq_train = rsq_train
                  ,rsq_test = rsq_test
                  ,nrmse_train = nrmse_train
                  ,nrmse_test = nrmse_test
                  ,mape_mod = mape_mod
                  ,coefs = coefs
                  ,y_pred = y_pred
                  ,mod=mod)
  
  return(mod_out)
}

################################################################
#### Define major mmm function

f.mmm <- function(...
                  , set_iter = 100
                  , set_cores = 6
                  , lambda.n = 100
                  , fixed.out = F
                  , optimizer_name = "DiscreteOnePlusOne" # c("DiscreteOnePlusOne", "DoubleFastGADiscreteOnePlusOne", "TwoPointsDE", "DE")
) {
  
  ################################################
  #### Collect hyperparameters
  
  #input.collect <- unlist(list(...), recursive = F) # input.collect <- set_hyperBoundLocal input.fixed <- dt_hyperResult fixed.out = T
  if (fixed.out==F) {
    input.collect <- unlist(list(...), recursive = F)
  } else {
    input.collect <- set_hyperBoundLocal
    input.fixed <- dt_hyperResult
  }
  hypParamSamName <- f.getHyperNames()
  
  # sort hyperparameter list by name
  hyper_bound_local <- list()
  for (i in 1:length(hypParamSamName)) {
    hyper_bound_local[i] <- input.collect[hypParamSamName[i]]
    names(hyper_bound_local)[i] <- hypParamSamName[i]
  }
  
  # get hyperparameters for Nevergrad
  bounds_ng <- which(sapply(hyper_bound_local, length)==2)
  hyper_bound_local_ng <- hyper_bound_local[bounds_ng]
  hyper_bound_local_ng_name <- names(hyper_bound_local_ng)
  num_hyppar_ng <- length(hyper_bound_local_ng)
  if (num_hyppar_ng == 0) {fixed.out <- T}
  
  # get fixed hyperparameters
  bounds_fixed <- which(sapply(hyper_bound_local, length)==1)
  hyper_bound_local_fixed <- hyper_bound_local[bounds_fixed]
  hyper_bound_local_fixed_name <- names(hyper_bound_local_fixed)  
  num_hyppar_fixed <- length(hyper_bound_local_fixed)
  
  #hyper_bound_local_fixed <- list(print_S_alphas = 1 , print_S_gammas = 0.5)
  if (set_cores >1) {
    hyper_bound_local_fixed_dt <- data.table(sapply(hyper_bound_local_fixed, function(x) rep(x, set_cores)))
  } else {
    hyper_bound_local_fixed_dt <- as.data.table(matrix(hyper_bound_local_fixed, nrow = 1))
    names(hyper_bound_local_fixed_dt) <- hyper_bound_local_fixed_name
  }
  
  # if (fixed.out == T) {
  #   hyperParams <- input.collect
  # }
  
  ####### We should definitely move this to the .exec script
  # Load reticulate to use it as a python interface for nevergrad:
  #library("reticulate")
  
  # reticulate::use_python("/Users/leonelsentana/Library/r-miniconda/envs/r-reticulate/bin/python")
  # reticulate::use_python("/Users/oteytaud/Library/r-miniconda/envs/robynng/bin/python")
  
  # Create and define r-reticulate conda environment to use
  # conda_create("r-reticulate")
  #use_condaenv("r-reticulate")
  
  # Install nevergrad package within conda:
  # conda_install("r-reticulate", "nevergrad", pip=TRUE)
  ####### We should definitely move this to the .exec script
  
  #Assign names to sampled hyperparameter vectors
  
  # assign("hyperparameters", hyperParams, envir = .GlobalEnv)
  
  # for (i in 1:length(hyperParams)) {  # This is a loop over the e.g. 50 000 hyperparam vectors
  #   assign(names(hyperParams)[i], hyperParams[[i]])
  # } #hyperParams <- mapply(FUN = function(x,y) {assign(y, x)}, x = hyperParams, y= names(hyperParams))
  
  ################################################
  #### Get spend share
  
  dt_spendShare <- dt_input[, .(rn = set_mediaVarName,
                                total_spend = sapply(.SD, sum)), .SDcols=set_mediaSpendName]
  dt_spendShare[, ':='(spend_share = total_spend / sum(total_spend))]
  
  ################################################
  #### Setup environment
  
  ## get environment for parallel backend
  dt_mod <- dt_mod
  set_mediaVarName <- set_mediaVarName
  adstock <- adstock
  set_modTrainSize <- set_modTrainSize
  activate_calibration <- activate_calibration
  set_baseVarSign <- set_baseVarSign
  set_mediaVarSign <- set_mediaVarSign
  activate_prophet <- activate_prophet
  set_prophetVarSign <- set_prophetVarSign
  set_factorVarName <- set_factorVarName
  set_lift <- set_lift
  
  ## set parallel backend
  #cl <- makeCluster(set_cores)# makeSOCKcluster(set_cores) #makeCluster(set_cores)
  # registerDoSNOW(cl)
  # getDoParWorkers()
  
  
  ################################################
  ng <- import("nevergrad")
  #### Start parallel loop
  t0 <- Sys.time()
  
  # Creating an optimizer. Use "none" for the good old LHS.
  # optimizer_name <- "none"  # good old LHS :-)
  # optimizer_name <- "DoubleFastGADiscreteOnePlusOne"
  # optimizer_name <- "OnePlusOne"
  # optimizer_name <- "DE"
  # optimizer_name <- "RandomSearch"
  # optimizer_name <- "TwoPointsDE"
  # optimizer_name <- "Powell"
  # optimizer_name <- "MetaModel"  CRASH !!!!
  # optimizer_name <- "SQP"
  # optimizer_name <- "Cobyla"
  # optimizer_name <- "NaiveTBPSA"
  # optimizer_name <- "DiscreteOnePlusOne"
  # optimizer_name <- "cGA"
  # optimizer_name <- "ScrHammersleySearch"
  
  # If optimizer_name is "none" then we take the hyperParams already in lhsOut <- f.hypSamLHS() 
  # on the "Collect hyperparameters" section above.
  # If optimizer_name is not "none" then we will create a vector of hyperparameters using Nevergrad (nevergrad_hp_val)
  
  ## Nevergrad parallel loop
  
  #loopNG <- ifelse(optimizer_name != "none", ceiling(iterRS / set_cores), 1)
  if (fixed.out == F) {
    iterTotal <- set_iter
    iterPar <- set_cores
  } else if (num_hyppar_ng==0 & fixed.out == T) {
    iterTotal <- 1
    iterPar <- 1
  } else {
    iterTotal <- nrow(input.fixed)
    iterPar <- nrow(input.fixed)
  }
  iterNG <-  ifelse(fixed.out == F, ceiling(set_iter/set_cores), 1)
  
  cat("\nRunning", iterTotal,"iterations with evolutionary algorithm on",adstock, "adstocking,", length(hyper_bound_local_ng),"hyperparameters,",lambda.n,"-fold ridge x-validation using",set_cores,"cores...\n")
  
  if (length(hyper_bound_local_ng) !=0) {
    my_tuple <- tuple(num_hyppar_ng)
    instrumentation <- ng$p$Array(shape=my_tuple)
    instrumentation$set_bounds(0., 1.)
    optimizer <-  ng$optimizers$registry[optimizer_name](instrumentation, budget=iterTotal, num_workers=set_cores)
    if (activate_calibration==F) {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1.0, 1.0))
    } else {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1.0, 1.0, 1.0))
    }
    # Creating a hyperparameter vector to be used in the next learning.
  }
  
  
  resultCollectNG <- list()
  cnt <- 0
  cat('\n',"Working with: ", optimizer_name,'\n')
  if(fixed.out==F) {pb <- txtProgressBar(max = iterTotal, style = 3)}
  #opts <- list(progress = function(n) setTxtProgressBar(pb, n))
  sysTimeDopar <- system.time({
    for (lng in 1:iterNG) {
      
      nevergrad_hp <- list()
      nevergrad_hp_val <- list()
      hypParamSamList <- list()
      hypParamSamNG <- c()
      
      if (fixed.out == F) {
        for (co in 1:iterPar) {
          nevergrad_hp[[co]] <- optimizer$ask()
          nevergrad_hp_val[[co]] <- nevergrad_hp[[co]]$value
          
          #print(paste0("LHS provides this:", hypParamSam,"nevergrad provides: ", nevergrad_hp_val, 
          # ", this is a pure vector, to be cast to some bounds and converted into a table, which is done below" ))
         
          #print(hypParamSamName)
          for (hypNameLoop in hyper_bound_local_ng_name) { # hypNameLoop <- local_name.all[1]
            #for (hypNameLoop in local_name.all) { # hypNameLoop <- local_name.all[1]
            index <- which(hypNameLoop == hyper_bound_local_ng_name)
            # get channel bounds
            # This should be done once and for all, not everytime. TODO
            
            #if (epoch.iter==1 | !identical(set_hyperBoundLocal, set_hyperBoundLocal.update))  { # Manual tuning, first epoch --> take manual params
            channelBound <- unlist(hyper_bound_local_ng[hypNameLoop])
            # } else { # manual tuning, not first epoch --> take auto-updated bounds
            #   channelBound <- unlist(hyperbound.local.auto[hypNameLoop])
            # }
            
            
            # adjust sampling to bounds
            # if (length(channelBound)==2 & channelBound[1] != channelBound[2]) {
            ####channelLHS <- unlist(initLHS[, hypNameLoop, with = F])
            hyppar_for_qunif <- nevergrad_hp_val[[co]][index]  # <--- bad syntax
            hyppar_scaled <- qunif(hyppar_for_qunif, min(channelBound), max(channelBound))  # <--- bad syntax
            # } else if (length(channelBound)==1) {
            #   xt <- channelBound
            # }
            hypParamSamNG[hypNameLoop] <- hyppar_scaled    # <--- bad syntax
          }
          hypParamSamList[[co]] <- transpose(data.table(hypParamSamNG))
          ##### previous code was: transLHS <- dcast.data.table(transLHS, index ~ vars, value.var = "xt")[, !"index"]
          #print("we get this:")
          #print(hypParamSam)
          #####################################
        }
        
        hypParamSamNG<- rbindlist(hypParamSamList)
        hypParamSamNG <- setnames(hypParamSamNG, names(hypParamSamNG), hyper_bound_local_ng_name)
        
        ## add fixed hyperparameters
        
        if (num_hyppar_fixed != 0) {
          hypParamSamNG <- cbind(hypParamSamNG, hyper_bound_local_fixed_dt)
          hypParamSamNG <- setcolorder(hypParamSamNG, hypParamSamName)
        }
      } else if (num_hyppar_ng==0 & fixed.out == T) {
        hypParamSamNG <- as.data.table(matrix(unlist(hyper_bound_local), nrow = 1))
        setnames(hypParamSamNG, names(hypParamSamNG), hypParamSamName)
      } else {
        hypParamSamNG <- input.fixed[, hypParamSamName, with = F]
      }
      
      
      
      ## Parallel start
      
      nrmse.collect <- c()
      decomp.rssd.collect <- c()
      best_mape <- Inf
      registerDoParallel(cores=set_cores)
      getDoParWorkers()
      doparCollect <- foreach (
        i = 1:iterPar
        , .export = c('f.adstockGeometric'
                      , 'f.adstockWeibull'
                      , 'f.transformation'
                      , 'f.rsq'
                      , 'f.decomp'
                      , 'f.calibrateLift'
                      , 'f.lambdaRidge'
                      , 'f.refit')
        , .packages = c('glmnet'
                        ,'stringr'
                        ,'data.table'
        )
        #, .options.snow = opts
      )  %dopar%  {
        
        t1 <- Sys.time()
        
        #####################################
        #### Get hyperparameter sample
        
        # Let us create a vector of hyperparameters using Nevergrad (nevergrad_hp_val).
        # Note: Maybe we should add LHS within the IF so that we skip running LHS if (optimizer_name != "none")?
        # hypParamSam <-  sapply(hyperParams, function(x) {if (length(x) > 1) { x[i] } else {x} }); hypParamSam
        
        hypParamSam <- unlist(hypParamSamNG[i])
        
        #### Tranform media with hyperparameters
        dt_modAdstocked <- dt_mod[, .SD, .SDcols = setdiff(names(dt_mod), "ds")]
        mediaAdstocked <- list()
        mediaVecCum <- list()
        for (v in 1:length(set_mediaVarName)) {
          
          m <- dt_modAdstocked[, get(set_mediaVarName[v])]
          
          if (adstock == "geometric") {
            theta = hypParamSam[paste0(set_mediaVarName[v],"_thetas")]
            alpha = hypParamSam[paste0(set_mediaVarName[v],"_alphas")]
            gamma = hypParamSam[paste0(set_mediaVarName[v],"_gammas")]
            mediaAdstocked[[v]] <- f.transformation(x=m, theta=theta, shape = shape, scale = scale, alpha=alpha, gamma=gamma, alternative = adstock)
            mediaVecCum[[v]] <- f.transformation(x=m, theta=theta, shape = shape, scale = scale, alpha=alpha, gamma=gamma, alternative = adstock, stage = "thetaVecCum")
            
          } else if (adstock == "weibull") {
            shape = hypParamSam[paste0(set_mediaVarName[v],"_shapes")]
            scale = hypParamSam[paste0(set_mediaVarName[v],"_scales")]
            alpha = hypParamSam[paste0(set_mediaVarName[v],"_alphas")]
            gamma = hypParamSam[paste0(set_mediaVarName[v],"_gammas")]
            mediaAdstocked[[v]] <- f.transformation(x=m, theta=theta, shape = shape, scale = scale, alpha=alpha, gamma=gamma, alternative = adstock)
            mediaVecCum[[v]] <- f.transformation(x=m, theta=theta, shape = shape, scale = scale, alpha=alpha, gamma=gamma, alternative = adstock, stage = "thetaVecCum")
          } else {break; print("adstock parameter must be geometric or weibull")}
        }
        
        names(mediaAdstocked) <- set_mediaVarName
        dt_modAdstocked[, (set_mediaVarName) := mediaAdstocked]
        dt_mediaVecCum <- data.table()[, (set_mediaVarName):= mediaVecCum]

        #####################################
        #### Split and prepare data for modelling
        
        trainSize <- round(nrow(dt_modAdstocked)* set_modTrainSize)
        dt_train <- dt_modAdstocked[1:trainSize]
        dt_test <- dt_modAdstocked[(trainSize+1):nrow(dt_modAdstocked)]
        
        ## contrast matrix because glmnet does not treat categorical variables
        y_train <- dt_train$depVar
        x_train <- model.matrix(depVar ~., dt_train)[, -1]
        y_test <- dt_test$depVar
        x_test <- model.matrix(depVar ~., dt_test)[, -1]
        y <- c(y_train, y_test)
        x <- rbind(x_train, x_test)
        
        ## create lambda sequence with x and y
        lambda_seq <- f.lambdaRidge(x=x_train, y=y_train, seq_len = lambda.n, lambda_min_ratio = 0.0001)
        
        ## define sign control
        dt_sign <- dt_modAdstocked[, !"depVar"] #names(dt_sign)
        #x_sign <- if (activate_prophet) {c(set_prophetVarSign, set_baseVarSign, set_mediaVarSign)} else {c(set_baseVarSign, set_mediaVarSign)}
        x_sign <- c(set_prophetVarSign, set_baseVarSign, set_mediaVarSign)
        check_factor <- sapply(dt_sign, is.factor)
        
        lower.limits <- c(); upper.limits <- c()
        
        for (s in 1:length(check_factor)) {
          
          if (check_factor[s]==T) {
            level.n <- length(levels(unlist(dt_sign[, s, with=F])))
            if (level.n <=1) {stop("factor variables must have more than 1 level")}
            lower_vec <- if(x_sign[s] == "positive") {rep(0, level.n-1)} else {rep(-Inf, level.n-1)}
            upper_vec <- if(x_sign[s] == "negative") {rep(0, level.n-1)} else {rep(Inf, level.n-1)}
            lower.limits <- c(lower.limits, lower_vec)
            upper.limits <- c(upper.limits, upper_vec)
          } else {
            lower.limits <- c(lower.limits, ifelse(x_sign[s] == "positive", 0, -Inf))
            upper.limits <- c(upper.limits ,ifelse(x_sign[s] == "negative", 0, Inf))
          }
        }
        
        
        
        #####################################
        #### fit ridge regression with x-validation
        cvmod <- cv.glmnet(x_train
                           ,y_train
                           ,family = "gaussian"
                           ,alpha = 0 #0 for ridge regression
                           #,lambda = lambda_seq
                           ,lower.limits = lower.limits
                           ,upper.limits = upper.limits
                           ,type.measure = "mse"
                           #,nlambda = 100
                           #,intercept = FALSE
        ) # plot(cvmod) coef(cvmod)
        
        
        
        #####################################
        #### refit ridge regression with selected lambda from x-validation

          
          ## if no lift calibration, refit using best lambda
          
          mod_out <- f.refit(x_train, y_train, x_test, y_test, lambda=cvmod$lambda.1se, lower.limits, upper.limits)
          lambda <- cvmod$lambda.1se
          #hypParamSam["lambdas"] <- cvmod$lambda.1se
          #hypParamSamName <- names(hypParamSam)
          
          decompCollect <- f.decomp(coefs=mod_out$coefs, dt_modAdstocked, x, y_pred=mod_out$y_pred, i)
          nrmse <- mod_out$nrmse_test
          mape <- 0
          
         if (activate_calibration == T) {
          
          ## if lift calibration, refit using sub lambda sequence with lower error to allow lift calibration
          # lambda_seq_calibrate <- lambda_seq[lambda_seq >= cvmod$lambda.min & lambda_seq <=cvmod$lambda.1se]
          # mape <- c()
          #for (l in 1:length(lambda_seq_calibrate)) {
            
            # mod_out <- f.refit(x_train, y_train, x_test, y_test, lambda=lambda_seq_calibrate[l], lower.limits, upper.limits)
            # decompCollect <- f.decomp(mod_out$coefs, dt_modAdstocked, x, mod_out$y_pred, i)
            liftCollect <- f.calibrateLift(decompCollect, set_lift)
            mape <- liftCollect[, mean(mape_lift)]
            #mape[l] <- mean(c(mod_out$mape_mod, mape_lift))
          #}
          
          # mape <- mape[which.min(mape)]
          # mod_out <- f.refit(x_train, y_train, x_test, y_test, lambda=lambda_seq_calibrate[which.min(mape)], lower.limits, upper.limits)
          # decompCollect <- f.decomp(mod_out$coefs, dt_modAdstocked, x, mod_out$y_pred, i)
          # liftCollect <- f.calibrateLift(decompCollect, set_lift)
          # 
          # nrmse <- mod_out$nrmse_test
          # lambda <- lambda_seq_calibrate[which.min(mape)]
          #hypParamSam["lambdas"] <- lambda_seq_calibrate[which.min(mape)]
          #hypParamSamName <- names(hypParamSam)
          
        }
        
        #####################################
        #### calculate multi-objectives for pareto optimality
        
        ## decomp objective: sum of squared distance between decomp share and spend share to be minimised
        dt_decompSpendDist <- decompCollect$xDecompAgg[rn %in% set_mediaVarName, .(rn, xDecompPerc)]
        dt_decompSpendDist <- dt_decompSpendDist[dt_spendShare[, .(rn, spend_share, total_spend)], on = "rn"]
        dt_decompSpendDist[, effect_share:= xDecompPerc/sum(xDecompPerc)]
        decomp.rssd <- dt_decompSpendDist[, sqrt(sum((effect_share-spend_share)^2))]
        
        ## adstock objective: sum of squared infinite sum of decay to be minimised? maybe not necessary
        dt_decaySum <- dt_mediaVecCum[,  .(rn = set_mediaVarName, decaySum = sapply(.SD, sum)), .SDcols = set_mediaVarName]
        adstock.ssisd <- dt_decaySum[, sum(decaySum^2)]
        
        ## calibration objective: not calibration: mse, decomp.rssd, if calibration: mse, decom.rssd, mape_lift
        
        #####################################
        #### Collect output
        
        resultHypParam <- data.table()[, (hypParamSamName):= lapply(hypParamSam[1:length(hypParamSamName)], function(x) x)]
        
        resultCollect <- list(
          resultHypParam = resultHypParam[, ':='(mape = mape
                                                 ,nrmse = nrmse
                                                 ,decomp.rssd = decomp.rssd
                                                 ,adstock.ssisd = adstock.ssisd
                                                 ,rsq_train = mod_out$rsq_train
                                                 ,rsq_test = mod_out$rsq_test
                                                 ,pos = prod(decompCollect$xDecompAgg$pos)
                                                 ,lambda=lambda
                                                 #,Score = -mape
                                                 ,Elapsed = as.numeric(difftime(Sys.time(),t1, units = "secs"))
                                                 ,ElapsedAccum = as.numeric(difftime(Sys.time(),t0, units = "secs"))
                                                 ,iterPar= i
                                                 ,iterNG = lng)],
          xDecompVec = if (fixed.out == T) {decompCollect$xDecompVec[, ':='(mape = mape
                                                                      ,nrmse = nrmse
                                                                      ,decomp.rssd = decomp.rssd
                                                                      ,adstock.ssisd = adstock.ssisd
                                                                      ,rsq_train = mod_out$rsq_train
                                                                      ,rsq_test = mod_out$rsq_test
                                                                      ,lambda=lambda
                                                                      ,iterPar= i
                                                                      ,iterNG = lng)]} else{NULL} ,
          xDecompAgg = decompCollect$xDecompAgg[, ':='(mape = mape
                                                       ,nrmse = nrmse
                                                       ,decomp.rssd = decomp.rssd
                                                       ,adstock.ssisd = adstock.ssisd
                                                       ,rsq_train = mod_out$rsq_train
                                                       ,rsq_test = mod_out$rsq_test
                                                       ,lambda=lambda
                                                       ,iterPar= i
                                                       ,iterNG = lng)] ,
          liftCalibration = if (activate_calibration) {liftCollect[, ':='(mape = mape
                                                                          ,nrmse = nrmse
                                                                          ,decomp.rssd = decomp.rssd
                                                                          ,adstock.ssisd = adstock.ssisd
                                                                          ,rsq_train = mod_out$rsq_train
                                                                          ,rsq_test = mod_out$rsq_test
                                                                          ,lambda=lambda
                                                                          ,iterPar= i
                                                                          ,iterNG = lng)] } else {NULL},
          decompSpendDist = dt_decompSpendDist[, ':='(mape = mape
                                                      ,nrmse = nrmse
                                                      ,decomp.rssd = decomp.rssd
                                                      ,adstock.ssisd = adstock.ssisd
                                                      ,rsq_train = mod_out$rsq_train
                                                      ,rsq_test = mod_out$rsq_test
                                                      ,lambda=lambda
                                                      ,iterPar= i
                                                      ,iterNG = lng)],
          mape.lift = mape,
          nrmse = nrmse,
          decomp.rssd = decomp.rssd,
          iterPar = i,
          iterNG = lng
          #,cvmod = cvmod
        )

        best_mape <- min(best_mape, mape)
        if (cnt == iterTotal) {
          print(" === ")
          print(paste0("Optimizer_name: ",optimizer_name, ";  Total_iterations: ", cnt, ";   best_mape: ",best_mape))
        }
        return(resultCollect)
      } # end dopar
      ## end parallel
      
      nrmse.coolect <- sapply(doparCollect, function(x) x$nrmse)
      decomp.rssd.coolect <- sapply(doparCollect, function(x) x$decomp.rssd)
      mape.lift.coolect <- sapply(doparCollect, function(x) x$mape.lift)
      
      if (fixed.out == F) {
        if (activate_calibration == F) {
          for (co in 1:iterPar) {
            optimizer$tell(nevergrad_hp[[co]], tuple(nrmse.coolect[co], decomp.rssd.coolect[co])) 
          }
        } else {
          for (co in 1:iterPar) {
            optimizer$tell(nevergrad_hp[[co]], tuple(nrmse.coolect[co], decomp.rssd.coolect[co], mape.lift.coolect[co])) 
          }
        }

        
      }
      
      resultCollectNG[[lng]] <- doparCollect
      cnt <- cnt + iterPar
      if(fixed.out==F) {setTxtProgressBar(pb, cnt)}
      
    } ## end NG loop
  }) # end system.time
  
  cat("\n Finished in",sysTimeDopar[3]/60,"mins\n")
  if(fixed.out==F) {close(pb)}
  registerDoSEQ(); getDoParWorkers()
  
  ## Get pareto optimal results when using nevergrad algorithms
  
  if (fixed.out == F) {
    pareto_results<-transpose(rbind(as.data.table(sapply(optimizer$pareto_front(997, subset="domain-covering", subset_tentatives=500), function(p) round(p$value[],4))),
                                    as.data.table(sapply(optimizer$pareto_front(997, subset="domain-covering", subset_tentatives=500), function(p) round(p$losses[],4)))))
    if (activate_calibration == F) {
      pareto_results_names<-setnames(pareto_results, c(hyper_bound_local_ng_name,"nrmse", "decomp.rssd") )
      pareto_results_ordered<-setorder(pareto_results_names, "nrmse", "decomp.rssd")
    } else {
      pareto_results_names<-setnames(pareto_results, c(hyper_bound_local_ng_name,"nrmse", "decomp.rssd", "mape.lift") )
      pareto_results_ordered<-setorder(pareto_results_names, "nrmse", "decomp.rssd", "mape.lift")
    }
    

    #print(pareto_results_ordered)
  } else {
    pareto_results_ordered <- NULL
  }


  
  #please_stop_here()
  #stopCluster(cl)
  # aggregate result
  
  #max.row <- ifelse(nrow(dt_mod)*iterRS>=1000000,1000000, nrow(dt_mod)*iterRS)  ## max row to avoid memory exceed
  resultCollect <- list(
    resultHypParam = rbindlist(lapply(resultCollectNG, function(x) {rbindlist(lapply(x, function(y) y$resultHypParam))}))[order(nrmse)],
    xDecompVec = if (fixed.out==T) {rbindlist(lapply(resultCollectNG, function(x) {rbindlist(lapply(x, function(y) y$xDecompVec))}))[order(nrmse, ds)]} else {NULL},
    xDecompAgg =   rbindlist(lapply(resultCollectNG, function(x) {rbindlist(lapply(x, function(y) y$xDecompAgg))}))[order(nrmse)],
    liftCalibration = if(activate_calibration) {rbindlist(lapply(resultCollectNG, function(x) {rbindlist(lapply(x, function(y) y$liftCalibration))}))[order(mape, liftMedia, liftStart)]} else {NULL},
    decompSpendDist = rbindlist(lapply(resultCollectNG, function(x) {rbindlist(lapply(x, function(y) y$decompSpendDist))}))[order(nrmse)],
    #mape = unlist(lapply(doparCollect, function(x) x$mape)),
    #iterRS = unlist(lapply(doparCollect, function(x) x$iterRS)),
    paretoFront= as.data.table(pareto_results_ordered)
    #,cvmod = lapply(doparCollect, function(x) x$cvmod)
  )
  resultCollect$iter <- length(resultCollect$mape)
  resultCollect$best.iter <- resultCollect$resultHypParam$iterRS[1]
  resultCollect$elapsed.min <- sysTimeDopar[3]/60
  resultCollect$resultHypParam[, ElapsedAccum:= ElapsedAccum - min(ElapsedAccum) + resultCollect$resultHypParam[which.min(ElapsedAccum), Elapsed]] # adjust accummulated time
  resultCollect$resultHypParam
  #print(optimizer_name)
  #print(" get ")
  #print(best_mape)
  #please_stop_here()
  
  return(list(#Score =  -resultCollect$mape[iterRS], # score for BO
    resultCollect = resultCollect
    ,hyperBoundNG = hyper_bound_local_ng
    ,hyperBoundFixed = hyper_bound_local_fixed))
}

#####################################
#### Define best model parameter collection function

f.getOptimParRS <- function(model_output, kurt.tuner = 0) {
  return(f.plotHyperBoundOptim(F, channelPlot = NULL, model_output = model_output, kurt.tuner = kurt.tuner))
}


f.mmmRobyn <- function(set_hyperBoundLocal
                       ,optimizer_name = set_hyperOptimAlgo
                       ,ng_trial = set_trial # ng_trial <- 3
                       ,set_cores = set_cores
                       ,plot_folder = "~/Documents/GitHub/plots") {
  
  t0 <- Sys.time()
  hyperparameter_fixed <- all(sapply(set_hyperBoundLocal, length)==1)
  
  if (!hyperparameter_fixed) {
    ng_out <- list()
    ng_algos <- optimizer_name #c("DoubleFastGADiscreteOnePlusOne", "DiscreteOnePlusOne", "TwoPointsDE", "DE")
    #ng_iters <- c(300)
    
    
    t0 <- Sys.time()
    for (optmz in ng_algos) {
      ng_collect <- list()
      model_output_collect <- list()
      # for (itrs in ng_iters) {
      #   set_iter <- itrs
      #   ng_trials <- list()
      # loop_trial <- ng_trial[which(ng_iters == itrs)]
      for (ngt in 1:ng_trial) { 
        
        if (activate_calibration == F) {
          cat("\nRunning trial nr.", ngt,"out of",ng_trial,"...\n")
        } else {
          cat("\nRunning trial nr.", ngt,"out of",ng_trial,"with calibration...\n")
          
        }
        # rm(model_output)
        model_output <- f.mmm(set_hyperBoundLocal
                              ,set_iter = set_iter
                              ,set_cores = set_cores
                              ,optimizer_name = optmz
        )
        
        model_output["trials"] <- ngt
        ng_collect[[ngt]] <- model_output$resultCollect$paretoFront[, ':='(trials=ngt, iters = set_iter, ng_optmz = optmz)]
        model_output_collect[[ngt]] <- model_output
        #model_output_pareto <- f.mmm(set_hyperBoundLocal, out = T)
      }
      ng_collect <- rbindlist(ng_collect)
      px <- low(ng_collect$nrmse) * low(ng_collect$decomp.rssd)
      ng_collect <- psel(ng_collect, px, top = nrow(ng_collect))[order(trials, nrmse)]
      # }
      
      ng_out[[which(ng_algos==optmz)]] <- ng_collect
    }
    ng_out <- rbindlist(ng_out)
    setnames(ng_out, ".level", "manual_pareto")
    # fwrite(ng_out, './paretoRes.csv')
    
    if (!exists("plot_folder_sub")) {
      plot_folder_sub <- format(Sys.time(), "%Y-%m-%d %H.%M")
      plotPath <- dir.create(file.path(plot_folder, plot_folder_sub))
    }

    ## plot Pareto front
    
    resultHypParam <- rbindlist(lapply(model_output_collect, function (x) x$resultCollect$resultHypParam[, trials:= x$trials] ))
    resultHypParam[, solID:= (paste(trials,iterNG, iterPar, sep = "_"))]
    mape_lift_quantile10 <- quantile(resultHypParam$mape, probs = 0.10)
    nrmse_quantile90 <- quantile(resultHypParam$nrmse, probs = 0.90)
    decomprssd_quantile90 <- quantile(resultHypParam$decomp.rssd, probs = 0.90)
    resultHypParam[, mape.qt10:= mape <= mape_lift_quantile10 & nrmse <= nrmse_quantile90 & decomp.rssd <= decomprssd_quantile90]
    
    resultHypParamPareto <- resultHypParam[mape.qt10==T]
    px <- low(resultHypParamPareto$nrmse) * low(resultHypParamPareto$decomp.rssd)
    resultHypParamPareto <- psel(resultHypParamPareto, px, top = nrow(resultHypParamPareto))[order(iterNG, iterPar, nrmse)]
    setnames(resultHypParamPareto, ".level", "robynPareto")
    
    setkey(resultHypParam,solID)
    setkey(resultHypParamPareto,solID)
    resultHypParam <- merge(resultHypParam,resultHypParamPareto[, .(solID, robynPareto)], all.x=TRUE)

    num_pareto123 <- resultHypParam[robynPareto %in% c(1,2,3), .N]
    
    cat("\nPlotting", num_pareto123,"pareto optimum models in to folder",paste0(plot_folder, "/", plot_folder_sub,"/"),"...\n")
    pbplot <- txtProgressBar(max = num_pareto123, style = 3)
    
    
    pParFront <- ggplot(data = resultHypParam, aes(x=nrmse, y=decomp.rssd, color = robynPareto)) +
      geom_point(size = 0.5) +
      #stat_smooth(data = resultHypParam, method = 'gam', formula = y ~ s(x, bs = "cs"), size = 0.2, fill = "grey100", linetype="dashed")+
      geom_line(data = resultHypParam[robynPareto ==1], aes(x=nrmse, y=decomp.rssd), colour = "coral4")+
      geom_line(data = resultHypParam[robynPareto ==2], aes(x=nrmse, y=decomp.rssd), colour = "coral3")+
      geom_line(data = resultHypParam[robynPareto ==3], aes(x=nrmse, y=decomp.rssd), colour = "coral")+
      scale_colour_gradient(low = "navyblue", high = "skyblue") +
    labs(title="Model selection",
           subtitle=paste0("2D Pareto front 1-3 with ",optimizer_name,", iterations = ", set_iter , " * ", ng_trial, " trials"),
           x="NRMSE",
           y="DECOMP.RSSD")
    
    print(pParFront)
    ggsave(paste0(plot_folder, "/", plot_folder_sub,"/", "pareto_front.png")
           , plot = pParFront
           , dpi = 800, width = 12, height = 7)
    
    ## plot hyperparameter sampling distribution
    
    resultHypParam.melted <- melt.data.table(resultHypParam[, c(local_name,"robynPareto"), with = F], id.vars = c("robynPareto"))
    
    pSamp <- ggplot(data = resultHypParam.melted,  aes( x = value, y=variable, color = variable, fill = variable) ) +
      geom_violin(alpha = .5, size = 0) +
      geom_point(size = 0.2) +
      theme(legend.position = "none") +
      labs(title="Model selection", 
           subtitle=paste0("Hyperparameter pareto sample distribution", ", iterations = ", set_iter, " * ", ng_trial, " trials"),
           x="Hyperparameter space",
           y="")
    print(pSamp)
    ggsave(paste0(plot_folder, "/", plot_folder_sub,"/", "hypersampling.png")
           , plot = pSamp
           , dpi = 800, width = 12, height = 7)
    
    
    ## plot Pareto solutions
    
    xDecompAgg <- rbindlist(lapply(model_output_collect, function (x) x$resultCollect$xDecompAgg[, trials:= x$trials]))
    xDecompAgg <- xDecompAgg[resultHypParam, robynPareto := i.robynPareto, on = c("iterNG", "iterPar", "trials")]
    xDecompAgg[, solID:= (paste(trials,iterNG, iterPar, sep = "_"))]
    
    decompSpendDist <- rbindlist(lapply(model_output_collect, function (x) x$resultCollect$decompSpendDist[, trials:= x$trials]))
    decompSpendDist <- decompSpendDist[resultHypParam, robynPareto := i.robynPareto, on = c("iterNG", "iterPar", "trials")]
    decompSpendDist[, solID:= (paste(trials,iterNG, iterPar, sep = "_"))]
    decompSpendDist <- decompSpendDist[xDecompAgg[rn %in% set_mediaVarName, .(rn, xDecompAgg, solID)], on = c("rn", "solID")]
    decompSpendDist[, roi := xDecompAgg/total_spend ]
    
    
    paretoFronts <- c(1,2,3)
    cnt <- 0
    for (pf in paretoFronts) {
      
      plotMediaShare <- decompSpendDist[robynPareto == pf]
      plotWaterfall <- xDecompAgg[robynPareto == pf]
      
      uniqueSol <- plotMediaShare[, unique(solID)]
      for (j in 1:length(uniqueSol)) {
        
        cnt <- cnt+1
        ## plot spend x effect share comparison
        plotMediaShareLoop <- plotMediaShare[solID == uniqueSol[j]]
        rsq_test_plot <- plotMediaShareLoop[, round(unique(rsq_test),4)]
        nrmse_plot <- plotMediaShareLoop[, round(unique(nrmse),4)]
        decomp_rssd_plot <- plotMediaShareLoop[, round(unique(decomp.rssd),4)]
        
        plotMediaShareLoop <- melt.data.table(plotMediaShareLoop, id.vars = c("rn", "nrmse", "decomp.rssd", "rsq_test" ), measure.vars = c("spend_share", "effect_share", "roi"))
        plotMediaShareLoop[, rn:= factor(rn, levels = sort(set_mediaVarName))]
        plotMediaShareLoopBar <- plotMediaShareLoop[variable %in% c("spend_share", "effect_share")]
        plotMediaShareLoopLine <- plotMediaShareLoop[variable =="roi"]
        ySecScale <- max(plotMediaShareLoopLine$value)/max(plotMediaShareLoopBar$value)*1.1
        
        p1 <- ggplot(plotMediaShareLoopBar, aes(x=rn, y=value, fill = variable)) +
          geom_bar(stat = "identity", width = 0.5, position = "dodge") +
          geom_text(aes(label=paste0(round(value*100,2),"%")), color = "darkblue",  position=position_dodge(width=0.5), fontface = "bold") +
          
          geom_line(data = plotMediaShareLoopLine, aes(x = rn, y=value/ySecScale, group = 1, color = variable), inherit.aes = FALSE) +
          geom_point(data = plotMediaShareLoopLine, aes(x = rn, y=value/ySecScale, group = 1, color = variable), inherit.aes = FALSE, size=4) +
          geom_text(data = plotMediaShareLoopLine, aes(label=round(value,2), x = rn, y=value/ySecScale, group = 1, color = variable)
                    , fontface = "bold", inherit.aes = FALSE, hjust = -1, size = 6) +
          scale_y_continuous(sec.axis = sec_axis(~.* ySecScale)) +          
          coord_flip() +
          theme( legend.title = element_blank(), legend.position = c(0.9, 0.2) ,axis.text.x = element_blank()) +
          scale_fill_brewer(palette = "Paired") +
          labs(title = "Share of Spend VS Share of Effect"
               ,subtitle = paste0("rsq_test: ", rsq_test_plot, 
                                  ", nrmse = ", nrmse_plot, 
                                  ", decomp.rssd = ", decomp_rssd_plot)
               ,y="", x="")
        
        ## plot waterfall
        plotWaterfallLoop <- plotWaterfall[solID == uniqueSol[j]][order(xDecompPerc)]
        plotWaterfallLoop[, end := cumsum(xDecompPerc)]
        plotWaterfallLoop[, end := 1-end]
        plotWaterfallLoop[, ':='(start =shift(end, fill = 1, type = "lag")
                                 ,id = 1:nrow(plotWaterfallLoop)
                                 ,rn = as.factor(rn)
                                 ,sign = as.factor(ifelse(xDecompPerc>=0, "pos", "neg")))]

        p2 <- suppressWarnings(ggplot(plotWaterfallLoop, aes(x= id, fill = sign)) +
                                 geom_rect(aes(x = rn, xmin = id - 0.45, xmax = id + 0.45, ymin = end, ymax = start), stat="identity") +
                                 scale_x_discrete("", breaks = levels(plotWaterfallLoop$rn), labels = plotWaterfallLoop$rn)+
                                 theme(axis.text.x = element_text(angle=65, vjust=0.6), legend.position = c(0.1, 0.1))  +
                                 geom_text(mapping = aes(label = paste0(f.unit_format(xDecompAgg),"\n", round(xDecompPerc*100, 2), "%")
                                                         ,y = rowSums(cbind(end,xDecompPerc/2))), fontface = "bold") +
                                 coord_flip() +
                                 labs(title="Response decomposition waterfall by predictor"
                                      ,subtitle = paste0("rsq_test: ", rsq_test_plot, 
                                                         ", nrmse = ", nrmse_plot, 
                                                         ", decomp.rssd = ", decomp_rssd_plot)
                                      ,x=""
                                      ,y=""))
        
        ## plot adstock rate
        
        resultHypParamLoop <- resultHypParam[solID == uniqueSol[j]]
        
        hypParam <- unlist(resultHypParamLoop[, local_name, with =F])
        dt_transformPlot <- dt_mod[, c("ds", set_mediaVarName), with =F] # independent variables
        dt_transformSpend <- cbind(dt_transformPlot[,.(ds)], dt_input[, c(set_mediaSpendName), with =F]) # spends of indep vars
        setnames(dt_transformSpend, names(dt_transformSpend), c("ds", set_mediaVarName))
        dt_transformSpendMod <- copy(dt_transformPlot) # prepare dt for 
        dt_transformAdstock <- copy(dt_transformPlot)
        dt_transformSaturation <- copy(dt_transformPlot)
        chnl_non_spend <- set_mediaVarName[!(set_mediaVarName==set_mediaSpendName)]
        
        m_decayRate <- list()
        if (adstock == "geometric") {
          for (med in 1:length(set_mediaVarName)) {
            
            med_select <- set_mediaVarName[med]
            # update non-spend variables
            if (med_select %in% chnl_non_spend) {
              sel_nls <- ifelse(modNLSCollect[channel == med_select, rsq_nls>rsq_lm],"nls","lm")
              dt_transformSpendMod[, (med_select):= yhatNLSCollect[channel==med_select & models == sel_nls, yhat]]
            }
            m <- dt_transformPlot[, get(med_select)]
            theta <- hypParam[paste0(set_mediaVarName[med], "_thetas")]
            alpha <- hypParam[paste0(set_mediaVarName[med], "_alphas")]
            gamma <- hypParam[paste0(set_mediaVarName[med], "_gammas")]
            dt_transformAdstock[, (med_select):= f.transformation(x=m, theta=theta, alpha=alpha, gamma=gamma, alternative = adstock, stage=1)] 
            dt_transformSaturation[, (med_select):= f.transformation(x=m, theta=theta, alpha=alpha, gamma=gamma, alternative = adstock, stage=3)] 

            m <- dt_transformPlot[, get(set_mediaVarName[med])]
            m_decayRate[[med]] <- data.table((f.transformation(x=m, theta=theta, alpha=alpha, gamma=gamma, alternative = adstock, stage="thetaVecCum")))
            
            setnames(m_decayRate[[med]], "V1", paste0(set_mediaVarName[med], "_decayRate"))
          }

        } else if (adstock == "weibull") {
          for (med in 1:length(set_mediaVarName)) {
            
            med_select <- set_mediaVarName[med]
            # update non-spend variables
            if (med_select %in% chnl_non_spend) {
              sel_nls <- ifelse(modNLSCollect[channel == med_select, rsq_nls>rsq_lm],"nls","lm")
              dt_transformSpendMod[, (med_select):= yhatNLSCollect[channel==med_select & models == sel_nls, yhat]]
            }
            m <- dt_transformPlot[, get(med_select)]
            shape <- hypParam[paste0(set_mediaVarName[med], "_shapes")]
            scale <- hypParam[paste0(set_mediaVarName[med], "_scales")]
            alpha <- hypParam[paste0(set_mediaVarName[med], "_alphas")]
            gamma <- hypParam[paste0(set_mediaVarName[med], "_gammas")]
            dt_transformAdstock[, (med_select):= f.transformation(x=m, shape=shape, scale=scale, alpha=alpha, gamma=gamma, alternative = adstock, stage=1)] 
            dt_transformSaturation[, (med_select):= f.transformation(x=m, shape=shape, scale=scale, alpha=alpha, gamma=gamma, alternative = adstock, stage=3)] 
            
            m <- dt_transformPlot[, get(set_mediaVarName[med])]
            m_decayRate[[med]] <- data.table((f.transformation(x=m, shape= shape, scale=scale, alpha=alpha, gamma=gamma, alternative = adstock, stage="thetaVecCum")))
            setnames(m_decayRate[[med]], "V1", paste0(set_mediaVarName[med], "_decayRate"))
          }
        }
        
        m_decayRate <- data.table(cbind(sapply(m_decayRate, function(x) sapply(x, function(y)y))))
        setnames(m_decayRate, names(m_decayRate), set_mediaVarName)
        m_decayRateSum <- m_decayRate[, lapply(.SD, sum), .SDcols = set_mediaVarName]
        
        decayRate.melt <- suppressWarnings(melt.data.table(m_decayRateSum))
        
        #decayRate.melt[, channel:=str_extract(decayRate.melt$variable, paste0(set_mediaVarName, collapse = "|"))]
        #decayRate.melt[, variable:=str_replace(decayRate.melt$variable, paste0(paste0(set_mediaVarName,"_"), collapse = "|"), "")]
        
        ## get geometric reference
        decayVec <- seq(0, 0.9, by = 0.001)
        decayInfSum <- c()
        for (i in 1:length(decayVec)) {
          decayInfSum[i] <- 1 / (1 - decayVec[i])-1
        }
        
        decayOut <- c()
        for (i in 1:nrow(decayRate.melt)) {
          decayOut[i] <- decayVec[which.min(abs(decayRate.melt$value[i] - decayInfSum))]
        }
        decayRate.melt[, avg_decay_rate:= decayOut]
        decayRate.melt[, variable:= factor(variable, levels = sort(set_mediaVarName))]
        
        p3 <- ggplot(decayRate.melt, aes(x=variable, y=avg_decay_rate, fill = "coral")) +
          geom_bar(stat = "identity", width = 0.5) +
          theme(legend.position = "none") +
          coord_flip() +
          geom_text(aes(label=paste0(round(avg_decay_rate*100,1), "%")),  position=position_dodge(width=0.5), fontface = "bold") +
          ylim(0,1) +
          labs(title = "Average adstock decay rate"
               ,subtitle = paste0("rsq_test: ", rsq_test_plot, 
                                  ", nrmse = ", nrmse_plot, 
                                  ", decomp.rssd = ", decomp_rssd_plot)
               ,y="", x="")
        
        
        
        ## plot response curve
        
        dt_transformSaturationDecomp <- copy(dt_transformSaturation)
        for (i in 1:length(set_mediaVarName)) {
          coef <- plotWaterfallLoop[rn == set_mediaVarName[i], coef]
          dt_transformSaturationDecomp[, (set_mediaVarName[i]):= .SD * coef, .SDcols = set_mediaVarName[i]]
        }
        
        #mediaAdstockFactorPlot <- dt_transformPlot[, lapply(.SD, sum), .SDcols = set_mediaVarName]  / dt_transformAdstock[, lapply(.SD, sum), .SDcols = set_mediaVarName]
        #dt_transformSaturationAdstockReverse <- data.table(mapply(function(x, y) {x*y},x= dt_transformAdstock[, set_mediaVarName, with=F], y= mediaAdstockFactorPlot))
        dt_transformSaturationSpendReverse <- copy(dt_transformAdstock)
        
        for (i in 1:length(set_mediaVarName)) {
          chn <- set_mediaVarName[i]
          if (chn %in% set_mediaVarName[costSelector]) {
            Vmax <- modNLSCollect[channel == chn, Vmax]
            Km <- modNLSCollect[channel == chn, Km]
            dt_transformSaturationSpendReverse[, (chn):=.SD * Km / (Vmax - .SD), .SDcols = chn] # reach to spend, reverse Michaelis Menthen: x = y*Km/(Vmax-y)
          } else if (chn %in% chnl_non_spend) {
            coef_lm <- modNLSCollect[channel == chn, coef_lm]
            dt_transformSaturationSpendReverse[, (chn):= .SD/coef_lm, .SDcols = chn] 
          } 
          # spendRatioFitted <- decompSpendDist[rn == chn, mean(total_spend)] / dt_transformSaturationSpendReverse[, sum(.SD), .SDcols = chn]
          # dt_transformSaturationSpendReverse[, (chn):= .SD * spendRatioFitted, .SDcols = chn]
        }
        
        dt_scurvePlot <- cbind(melt.data.table(dt_transformSaturationDecomp, id.vars = "ds", variable.name = "channel",value.name = "response"),
                               melt.data.table(dt_transformSaturationSpendReverse, id.vars = "ds", value.name = "spend")[, .(spend)]) 
        
        
        dt_scurvePlotMean <- dt_transformSpend[, !"ds"][, lapply(.SD, mean), .SDcols = set_mediaVarName]
        dt_scurvePlotMean <- melt.data.table(dt_scurvePlotMean, measure.vars = set_mediaVarName, value.name = "mean_spend", variable.name = "channel")
        dt_scurvePlotMean[, ':='(mean_response=0, next_dollar_response=0)]
        
        for (med in 1:length(set_mediaVarName)) {
          m <- dt_transformSaturationSpendReverse[, get(set_mediaVarName[med])]
          alpha <- hypParam[which(paste0(set_mediaVarName[med], "_alphas")==names(hypParam))]
          gamma <- hypParam[which(paste0(set_mediaVarName[med], "_gammas")==names(hypParam))]
          gammaTrans <- round(quantile(seq(range(m)[1], range(m)[2], length.out = 100), gamma),4)
          get_spend <- dt_scurvePlotMean[channel == set_mediaVarName[med], mean_spend]
          get_response <-  get_spend**alpha / (get_spend**alpha + gammaTrans**alpha)
          get_response_marginal <- (get_spend+1)**alpha / ((get_spend+1)**alpha + gammaTrans**alpha)
          coef <- plotWaterfallLoop[rn == set_mediaVarName[med], coef]
          dt_scurvePlotMean[channel == set_mediaVarName[med], mean_response := get_response * coef]
          dt_scurvePlotMean[channel == set_mediaVarName[med], next_dollar_response := get_response_marginal * coef - mean_response]
          
        }


        p4 <- ggplot(data= dt_scurvePlot, aes(x=spend, y=response, color = channel)) +
          geom_line() +
          geom_point(data = dt_scurvePlotMean, aes(x=mean_spend, y=mean_response, color = channel)) +
          geom_text(data = dt_scurvePlotMean, aes(x=mean_spend, y=mean_response,  label = round(mean_spend,0)), show.legend = F, hjust = -0.2)+
          theme(legend.position = c(0.9, 0.2)) +
          labs(title="Response curve and mean spend by channel"
               ,subtitle = paste0("rsq_test: ", rsq_test_plot, 
                                  ", nrmse = ", nrmse_plot, 
                                  ", decomp.rssd = ", decomp_rssd_plot)
               ,x="Spend" ,y="response")

        
        
        modID <- paste0("Model one-pager, on pareto front ", pf,", ID: ", uniqueSol[j])

        pg <- arrangeGrob(p1,p2,p3, p4, ncol=2, top = text_grob(modID, size = 15, face = "bold"))
        #grid.draw(pg)
        ggsave(filename=paste0(plot_folder, "/", plot_folder_sub,"/", uniqueSol[j],".png")
               , plot = pg
               , dpi = 900, width = 18, height = 12)
        
        setTxtProgressBar(pbplot, cnt)
        
      }
    }
  } else {
    
    model_output_fixed <- f.mmm(set_hyperBoundLocal
                          ,set_iter = 1
                          ,set_cores = 1
                          ,optimizer_name = optimizer_name
    )
    
    print(model_output_fixed$resultCollect$xDecompAgg)
  }
  
  cat("\nTotal time: ",difftime(Sys.time(),t0, units = "mins"), "mins\n")
  
  return(list(resultHypParam=resultHypParam,
              xDecompAgg=xDecompAgg,
              resultHypParam=resultHypParam,
              model_output_fixed=if(exists("model_output_fixed")) {model_output_fixed} else {NULL}))
  
  
  # for (its in ng_iters) {
  #   loop_trial <- ng_trial[which(ng_iters == its)]
  # for (los in c("nrmse", "decomp.rssd")) {
  # 
  #   ng_out_plot <- copy(ng_out)
  #   ng_out_med <- ng_out_plot[, .SD, .SDcols = c(los, "ng_optmz")]
  #   setnames(ng_out_plot, los, "loss"); setnames(ng_out_med, los, "loss")
  #   ng_out_med <- ng_out_med[, .(xMaxDen= density(loss)$x[which.max(density(loss)$y)],
  #                                yMaxDen= which.max(density(loss)$y)), by = "ng_optmz"]
  # 
  # 
  #   # print(ggplot(data = ng_out_plot) +
  #   #         stat_density(geom = "line", aes(x=loss), adjust = 1) +
  #   #         facet_wrap(~ng_optmz, scales = "free") +
  #   #         labs(title="Nevergrad performance",
  #   #              subtitle=paste0("loss = ", los, ", iterations = ", set_iter , " * ", ng_trial, " trials"),
  #   #              x=toupper(los),
  #   #              y="Density")+
  #   #         xlim(0, 1) +
  #   #         geom_vline(data= ng_out_med, mapping = aes(xintercept = xMaxDen), colour="blue")+
  #   #         geom_text(data = ng_out_med, aes(x = xMaxDen, y = 0, angle = 90, vjust = -0.5, hjust= -1, label = paste0("mode: ",round(xMaxDen,4))), colour="blue")
  #   # )
  # }
  # #}
  # 
  # #png(paste0(plot_folder, "/", plot_folder_sub,"/", "pareto_front.png"))
  # # print(ggplot(data = ng_out, aes(x=nrmse, y=decomp.rssd,  color = ng_optmz)) +
  # #         geom_point(size = 0.5) +
  # #         stat_smooth(data = ng_out, method = 'gam', formula = y ~ s(x, bs = "cs"), size = 0.2, fill = "grey100", linetype="dashed")+
  # #         geom_line(data = ng_out[ manual_pareto ==1])+
  # #         labs(title="Nevergrad performance",
  # #              subtitle=paste0("2D Pareto front, iterations = ", set_iter , " * ", ng_trial, " trials"),
  # #              x="NRMSE",
  # #              y="DECOMP.RSSD")
  # # )
  # #dev.off()
  # 
  # 
  # # get hyperparameters for Nevergrad
  # get_plot_name <- names(ng_out_plot)[str_detect(names(ng_out_plot), paste(set_mediaVarName, collapse = "|"))]
  # ng_out_plot_melted <- melt.data.table(ng_out_plot[, c(get_plot_name, "trials", "iters", "ng_optmz", "manual_pareto"), with = F], id.vars = c("trials", "iters", "ng_optmz", "manual_pareto"))
  # 
  # print(ggplot(data = ng_out_plot_melted,  aes( x = value, y=variable, color = variable, fill = variable) ) +
  #         geom_violin(alpha = .5, size = 0) +
  #         geom_point(size = 0.2) +
  #         facet_wrap(~ng_optmz, scales = "free") +
  #         theme(legend.position = "none") +
  #         labs(title="Nevergrad performance", 
  #              subtitle=paste0("Hyperparameter pareto sample distribution", ", iterations = ", set_iter, " * ", ng_trial, " trials"),
  #              x="Hyperparameter space",
  #              y="")+
  #         xlim(0, 1))
  # ggsave(paste0(plot_folder, "/", plot_folder_sub,"/", "hypersampling.png"), dpi = 800, width = 12, height = 7)
  # 
  # #}
  # 
  # difftime(Sys.time(),t0, units = "mins")
  # 
  # ## get result stats
  # ng_out_pareto1 <- ng_out[manual_pareto==1]
  # numRep <- nrow(ng_out_pareto1)
  # NGBounds <- model_output$hyperBoundNG
  # NGLocalName <- names(model_output$hyperBoundNG)
  # 
  # hypparCollect <- list()
  # for (i in 1:numRep) {
  #   hypparCollectLoop <- c()
  #   for (j in 1:length(NGLocalName)) {
  #     channelBound <- unlist(NGBounds[NGLocalName[j]])
  #     hyppar_for_qunif <- unlist(ng_out_pareto1[i, NGLocalName[j], with =F])
  #     hypparCollectLoop[j] <- qunif(hyppar_for_qunif, min(channelBound), max(channelBound))
  #   }
  #   hypparCollect[[i]] <- transpose(data.table(hypparCollectLoop))
  # }
  # dt_hyperResult <- rbindlist(hypparCollect)
  # setnames(dt_hyperResult, names(dt_hyperResult), NGLocalName)
  # 
  # # combine fixed hyperparameters
  # fixedLocalName <- names(model_output$hyperBoundFixed)
  # numCol <- length(fixedLocalName)
  # #getFixedLocalHyperWhich <- which(sapply(set_hyperBoundLocal, length)==1)
  # 
  # if (numCol!=0) {
  #   fixedLocalBounds <- model_output$hyperBoundFixed
  #   dt_fixedLocalBounds <- as.data.table(matrix(unlist(rep(fixedLocalBounds, numRep)), nrow = numRep, ncol = numCol, byrow = T))
  #   setnames(dt_fixedLocalBounds, names(dt_fixedLocalBounds), fixedLocalName)
  #   dt_hyperResult <- cbind(dt_hyperResult, dt_fixedLocalBounds)
  # }
  # 
  # setcolorder(dt_hyperResult, c(local_name))
  # 
  # model_output_pareto <- f.mmm(set_hyperBoundLocal, fixed.out = T)
  # model_output_pareto$resultCollect$decompSpendDist
  # model_output_pareto$resultCollect$resultHypParam[, .(nrmse, decomp.rssd, iterPar, iterNG)]
  # ng_out[manual_pareto==1, .(manual_pareto, nrmse, decomp.rssd)][order(nrmse)]
  
  
}
