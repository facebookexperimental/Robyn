# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################
###### Data transformation and helper functions
########################################################################

################################################################
#### Define basic condition check function

f.checkConditions <- function(dt_transform) {
  
  if (set_iter <= 50000) {message("we recommend to run at least 50k iterations per epoch at the beginning")}

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
  }
  
  if((adstock %in% c("geometric", "weibull")) == F) {stop("adstock must be 'geometric' or 'weibull'")}
  
  if(activate_hyperBoundLocalTuning ==T & all(str_detect(names(set_hyperBoundLocal) ,paste0(local_name, collapse = "|")))==F) {
    stop("check if set_hyperBoundLocal contains correct hyperparameters. also naming is case sensitive")
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
    stop("input date variable does should have format '2020-01-01'")
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
  
  if (!activate_hyperBoundLocalTuning) {
    assign("set_hyperBoundLocal", NULL, envir = .GlobalEnv)
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
            warning("default start value for nls out of range. using c(1,1) instead")
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
                                                             rsq_lm = rsq_lm
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
  if (adstock == "geometric") {
    local_name <- sort(apply(expand.grid(set_mediaVarName, global_name[global_name %like% 'thetas|alphas|gammas']), 1, paste, collapse="_"))
  } else if (adstock == "weibull") {
    local_name <- sort(apply(expand.grid(set_mediaVarName, global_name[global_name %like% 'shapes|scales|alphas|gammas']), 1, paste, collapse="_"))
  }
  return(local_name)
}


################################################
#### Define latin hypercube sampling function

f.hypSamLHS <- function(set_mediaVarName, set_iter, hyper_bound_global, adstock) {
  
  # translate global to local hyperparameters
  global_name <- names(hyper_bound_global)
  #set_mediaVarName <- toupper(set_mediaVarName)
  if (adstock == "geometric") {
    local_name.all <- sort(apply(expand.grid(set_mediaVarName, global_name[global_name %like% 'thetas|alphas|gammas']), 1, paste, collapse="_"))
  } else if (adstock == "weibull") {
    local_name.all <- sort(apply(expand.grid(set_mediaVarName, global_name[global_name %like% 'shapes|scales|alphas|gammas']), 1, paste, collapse="_"))
  } else {break; print("adstock must be geometric or weibull")}
  
  # check if any bounds are fixed
  if (activate_hyperBoundLocalTuning==T) {
    
    if (!exists("bounds_whichfixed") & epoch.iter>1) {
      stop("You have just changed to manual tuning mode. Please rerun the whole script")
    }
    
    set_hyperBoundLocalLen <- sapply(set_hyperBoundLocal, length)
    set_hyperBoundLocalLen <- set_hyperBoundLocalLen[order(names(set_hyperBoundLocalLen))]
    
    if ((length(set_hyperBoundLocalLen) != length(local_name.all)) | !all(set_hyperBoundLocalLen %in% c(1,2))) {
      stop("set_hyperBoundLocal must contain all hyperparameters correctly. Every parameter can contain 2 values as lower and upper bound or 1 value as fixed value.")
    }
    
    if (all(sapply(set_hyperBoundLocal, length)==1)) {stop("all set_hyperBoundLocal hyperparameters are fixed. run f.mmmCollect(set_hyperBoundLocal) to get final result")}
    
    bounds_whichfixed <- set_hyperBoundLocalLen==1
    local_name.update <- local_name.all[!bounds_whichfixed]
    
    # if start with tuning and  change parameters again
    if (epoch.iter==1) {
      assign("set_hyperBoundLocal.update", set_hyperBoundLocal, envir = .GlobalEnv)
    } else {
      if (!identical(set_hyperBoundLocal, set_hyperBoundLocal.update)) {
        stop("set_hyperBoundLocal has been changed. Please rerun the whole script")
      }
    }
    
    assign("bounds_whichfixed", bounds_whichfixed, envir = .GlobalEnv)
  } else {
    local_name.update <- local_name.all
  }
  assign("local_name.update", local_name.update, envir = .GlobalEnv)
  
  # generate random latin hypercube sampling 
  initLHS <- data.table(randomLHS(set_iter,length(local_name.update)))
  names(initLHS) <- local_name.update
  
  # rescale local bounds
  transLHS_collect <- list()
  for (hypNameLoop in local_name.all) { # hypNameLoop <- local_name.all[1]
    
    # get channel bounds
    if (activate_hyperBoundLocalTuning ==F | !exists("activate_hyperBoundLocalTuning")) {
      if (epoch.iter==1) { # no manual tuning, first epoch --> take global bounds
        channelBound <- unlist(hyper_bound_global[str_match(hypNameLoop, paste0(global_name, collapse = "|"))])
      } else { # no manual tuning, not first epoch --> take auto-updated bounds
        channelBound <- unlist(hyperbound.local.auto[hypNameLoop])
      }
    } else { 
      if (epoch.iter==1 | !identical(set_hyperBoundLocal, set_hyperBoundLocal.update))  { # Manual tuning, first epoch --> take manual params
        channelBound <- unlist(set_hyperBoundLocal[hypNameLoop])
      } else { # manual tuning, not first epoch --> take auto-updated bounds
        channelBound <- unlist(hyperbound.local.auto[hypNameLoop])
      }
    }
    
    # adjust sampling to bounds
    if (length(channelBound)==2 & channelBound[1] != channelBound[2]) {
      channelLHS <- unlist(initLHS[, hypNameLoop, with = F])
      xt <- qunif(channelLHS, min(channelBound), max(channelBound))
    } else {
      xt <- rep(set_hyperBoundLocal[[hypNameLoop]], set_iter)
    } 
    #xt <- min(channelBound) + qexp(channelLHS)/10* (max(channelBound) - min(channelBound))
    
    transLHS_collect[[hypNameLoop]] <- data.table(index = 1:set_iter, xt = xt, vars = hypNameLoop)
  }
  
  transLHS <- rbindlist(transLHS_collect)
  transLHS <- dcast.data.table(transLHS, index ~ vars, value.var = "xt")[, !"index"]
  
  transLHS.list <- lapply(transLHS, function(x) x)
  lhsOut <- list(transLHS.list=transLHS.list, initLHS=initLHS, transLHS=transLHS)
  return(lhsOut)
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
      
      ## scale decomp
      mmmDays <- nrow(liftPeriodVec) * 7 
      liftDays <- as.integer(liftEnd- liftStart + 1)
      y_hatLift <- sum(unlist(getDecompVec[, -1])) # total pred sales
      x_decompLift <- sum(liftPeriodVec[,2])
      x_decompLiftScaled <- x_decompLift / mmmDays * liftDays
      
      ## output
      liftCollect2[[lw]] <- data.table(liftMedia = getLiftMedia[m] ,
                                       liftStart = liftStart,
                                       liftEnd = liftEnd,
                                       liftAbs = set_lift[liftWhich[lw], liftAbs],
                                       decompAbsScaled = x_decompLiftScaled)
    }
    liftCollect[[m]] <- rbindlist(liftCollect2)
  }
  
  ## get mape_lift
  liftCollect <- rbindlist(liftCollect)[, mape_lift := abs((decompAbsScaled - liftAbs) / liftAbs) * 100]
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
  
  mod_out <- list(rsq_train = rsq_train
                  ,rsq_test = rsq_test
                  ,mape_mod = mape_mod
                  ,coefs = coefs
                  ,y_pred = y_pred
                  ,mod=mod)
  
  return(mod_out)
}

################################################################
#### Define major mmm function

f.mmm <- function(...
                  , iterRS = 1
                  , set_cores = 1
                  , lambda.n = 100
                  , out = F
) {
  
  ################################################
  #### Collect hyperparameters
  
  hyperParams.global <- unlist(list(...), recursive = F) # hyperParams.global <- set_hyperBoundGlobal 
  
  if (out == F) {
    lhsOut <- f.hypSamLHS(set_mediaVarName, set_iter = iterRS, hyperParams.global, adstock)
    hyperParams <- lhsOut$transLHS.list
  } else {
    hyperParams <- hyperParams.global
  }
  
  assign("hyperparameters", hyperParams, envir = .GlobalEnv)
  
  for (i in 1:length(hyperParams)) {
    assign(names(hyperParams)[i], hyperParams[[i]])
  } #hyperParams <- mapply(FUN = function(x,y) {assign(y, x)}, x = hyperParams, y= names(hyperParams))
  
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

  ## set paralle backend
  cat("\nRunning", iterRS,"random search trails with",lambda.n,"trails lambda cross-validation each on",set_cores,"cores...\n")
  cl <- makeCluster(set_cores)# makeSOCKcluster(set_cores) #makeCluster(set_cores)
  registerDoSNOW(cl)
  pb <- txtProgressBar(max = iterRS, style = 3)
  opts <- list(progress = function(n) setTxtProgressBar(pb, n))

  getDoParWorkers()
  
  ################################################
  #### Start parallel loop
  t0 <- Sys.time()
  sysTimeDopar <- system.time({
    doparCollect <- foreach (
      i = 1:iterRS
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
      , .options.snow = opts
    )  %dopar%  {
      
      t1 <- Sys.time()
      
      #####################################
      #### Get hyperparameter sample
      
      hypParamSam <-  sapply(hyperParams, function(x) {if (length(x) > 1) { x[i] } else {x} }); hypParamSam
      hypParamSamName <- names(hypParamSam)
      
      #####################################
      #### Tranform media with hyperparameters
      dt_modAdstocked <- dt_mod[, .SD, .SDcols = setdiff(names(dt_mod), "ds")]
      mediaAdstocked <- list()
      mediaVecCum <- list()
      for (v in 1:length(set_mediaVarName)) {
        
        m <- dt_modAdstocked[, get(set_mediaVarName[v])]
        
        if (adstock == "geometric") {
          theta = hypParamSam[str_which(hypParamSamName, paste0(set_mediaVarName[v],".*thetas"))]
          alpha = hypParamSam[str_which(hypParamSamName, paste0(set_mediaVarName[v],".*alphas"))] 
          gamma = hypParamSam[str_which(hypParamSamName, paste0(set_mediaVarName[v],".*gammas"))] 
          mediaAdstocked[[v]] <- f.transformation(x=m, theta=theta, shape = shape, scale = scale, alpha=alpha, gamma=gamma, alternative = adstock)
          mediaVecCum[[v]] <- f.transformation(x=m, theta=theta, shape = shape, scale = scale, alpha=alpha, gamma=gamma, alternative = adstock, stage = "thetaVecCum")
          
        } else if (adstock == "weibull") {
          shape = hypParamSam[str_which(hypParamSamName, paste0(set_mediaVarName[v],".*shapes"))]
          scale = hypParamSam[str_which(hypParamSamName, paste0(set_mediaVarName[v],".*scales"))]
          alpha = hypParamSam[str_which(hypParamSamName, paste0(set_mediaVarName[v],".*alphas"))] 
          gamma = hypParamSam[str_which(hypParamSamName, paste0(set_mediaVarName[v],".*gammas"))] 
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
                         ,lambda = lambda_seq
                         ,lower.limits = lower.limits
                         ,upper.limits = upper.limits
                         ,type.measure = "mse"
                         #,nlambda = 100
                         #,intercept = FALSE
      ) # plot(cvmod) coef(cvmod)
      
      
      
      #####################################
      #### refit ridge regression with selected lambda from x-validation
      if (activate_calibration == F) {
        
        ## if no lift calibration, refit using best lambda
        
        mod_out <- f.refit(x_train, y_train, x_test, y_test, lambda=cvmod$lambda.1se, lower.limits, upper.limits)
        
        hypParamSam["lambdas"] <- cvmod$lambda.1se
        hypParamSamName <- names(hypParamSam)
        
        decompCollect <- f.decomp(coefs=mod_out$coefs, dt_modAdstocked, x, y_pred=mod_out$y_pred, i)
        
        mape <- mod_out$mape_mod
        
      } else {
        
        ## if lift calibration, refit using sub lambda sequence with lower error to allow lift calibration
        
        lambda_seq_calibrate <- lambda_seq[lambda_seq >= cvmod$lambda.min & lambda_seq <=cvmod$lambda.1se]
        
        mape <- c()
        for (l in 1:length(lambda_seq_calibrate)) {
          
          mod_out <- f.refit(x_train, y_train, x_test, y_test, lambda=lambda_seq_calibrate[l], lower.limits, upper.limits) 
          decompCollect <- f.decomp(mod_out$coefs, dt_modAdstocked, x, mod_out$y_pred, i)
          liftCollect <- f.calibrateLift(decompCollect, set_lift)
          mape_lift <- liftCollect[, mean(mape_lift)]
          mape[l] <- mean(c(mod_out$mape_mod, mape_lift))
        }
        
        mape <- mape[which.min(mape)]
        mod_out <- f.refit(x_train, y_train, x_test, y_test, lambda=lambda_seq_calibrate[which.min(mape)], lower.limits, upper.limits) 
        decompCollect <- f.decomp(mod_out$coefs, dt_modAdstocked, x, mod_out$y_pred, i)
        liftCollect <- f.calibrateLift(decompCollect, set_lift)
        
        hypParamSam["lambdas"] <- lambda_seq_calibrate[which.min(mape)]
        hypParamSamName <- names(hypParamSam)
        
      } 
      
      #####################################
      #### calculate multi-objectives for pareto optimality
      
      ## decomp objective: sum of squared distance between decomp share and spend share to be minimised
      dt_decompSpendDist <- decompCollect$xDecompAgg[rn %in% set_mediaVarName, .(rn, xDecompPerc)]
      dt_decompSpendDist <- dt_decompSpendDist[dt_spendShare[, .(rn, spend_share)], on = "rn"]
      decomp.ssd <- dt_decompSpendDist[, sum((xDecompPerc-spend_share)^2) ] 
      
      ## adstock objective: sum of squared infinite sum of decay to be minimised
      dt_decaySum <- dt_mediaVecCum[,  .(rn = set_mediaVarName, decaySum = sapply(.SD, sum)), .SDcols = set_mediaVarName]
      adstock.ssisd <- dt_decaySum[, sum(decaySum^2)]
      
      ## saturation objective:

      
      #####################################
      #### Collect output
      
      resultHypParam <- data.table()[, (hypParamSamName):= lapply(hypParamSam, function(x) x)]
      
      resultCollect <- list(
        resultHypParam = resultHypParam[, ':='(mape = mape
                                               ,decomp.ssd = decomp.ssd
                                               ,adstock.ssisd = adstock.ssisd
                                               ,rsq_test = mod_out$rsq_test
                                               ,pos = prod(decompCollect$xDecompAgg$pos)
                                               ,Score = -mape
                                               ,Elapsed = as.numeric(difftime(Sys.time(),t1, units = "secs"))
                                               ,ElapsedAccum = as.numeric(difftime(Sys.time(),t0, units = "secs"))
                                               ,iterRS= i)],
        xDecompVec = if (out ==T) {decompCollect$xDecompVec[, ':='(mape = mape
                                                                   ,decomp.ssd = decomp.ssd
                                                                   ,adstock.ssisd = adstock.ssisd
                                                                   ,rsq_test = mod_out$rsq_test
                                                                   ,iterRS= i)]} else{NULL} ,
        xDecompAgg = decompCollect$xDecompAgg[, ':='(mape = mape
                                                     ,decomp.ssd = decomp.ssd
                                                     ,adstock.ssisd = adstock.ssisd
                                                     ,rsq_test = mod_out$rsq_test
                                                     ,iterRS= i)] ,
        liftCalibration = if (activate_calibration) {liftCollect[, ':='(mape = mape
                                                                        ,decomp.ssd = decomp.ssd
                                                                        ,adstock.ssisd = adstock.ssisd
                                                                        ,rsq_test = mod_out$rsq_test
                                                                        ,iterRS= i)] } else {NULL},
        mape = mape,
        iterRS = i
        #,cvmod = cvmod
      )
      
      setTxtProgressBar(pb, i)
      
      return(resultCollect)
    } # end dopar
  }) # end system.time
  ## end multicore
  
  cat("\ndone for", iterRS,"random search trails in",sysTimeDopar[3]/60,"mins")
  close(pb)
  stopCluster(cl)
  
  registerDoSEQ(); getDoParWorkers()
  
  # aggregate result
  
  max.row <- ifelse(nrow(dt_mod)*iterRS>=1000000,1000000, nrow(dt_mod)*iterRS)  ## max row to avoid memory exceed
  resultCollect <- list(
    resultHypParam = rbindlist(lapply(doparCollect, function(x) x$resultHypParam))[order(mape)],
    xDecompVec = if (out==T) {rbindlist(lapply(doparCollect, function(x) x$xDecompVec))[order(mape, ds)][1:max.row]} else {NULL},
    xDecompAgg = rbindlist(lapply(doparCollect, function(x) x$xDecompAgg))[order(mape)],
    liftCalibration = if(activate_calibration) {rbindlist(lapply(doparCollect, function(x) x$liftCalibration))[order(mape, liftMedia, liftStart)]} else {NULL},
    mape = unlist(lapply(doparCollect, function(x) x$mape)),
    iterRS = unlist(lapply(doparCollect, function(x) x$iterRS))
    #,cvmod = lapply(doparCollect, function(x) x$cvmod) 
  )
  resultCollect$iter <- length(resultCollect$mape)
  resultCollect$best.iter <- resultCollect$resultHypParam$iterRS[1]
  resultCollect$elapsed.min <- sysTimeDopar[3]/60
  resultCollect$resultHypParam[, ElapsedAccum:= ElapsedAccum - min(ElapsedAccum) + resultCollect$resultHypParam[which.min(ElapsedAccum), Elapsed]] # adjust accummulated time
  
  return(list(Score =  -resultCollect$mape[iterRS] # score for BO
              ,resultCollect = resultCollect))
}

#####################################
#### Define best model parameter collection function

f.getOptimParRS <- function(model_output, kurt.tuner = 0) {
  return(f.plotHyperBoundOptim(F, channelPlot = NULL, model_output = model_output, kurt.tuner = kurt.tuner))
}

#####################################
#### Define Robyn (Random-search and kurtOsis Based hYperparameter optimiesatioN) function

f.mmmRobyn <- function(hyper_bound_global = set_hyperBoundGlobal
                          ,set_iter = set_iter
                          ,set_cores = set_cores
                          ,epochN = Inf
                          ,out = F
                          ,temp.csv.path
                          ,optim.sensitivity = 0
) {
  
  assign("set_iter", set_iter, envir = .GlobalEnv)
  
  # hyper optimisation loop
  optim.loop <- T
  optim.iter <- 1
  if (!exists("model_output")) {
    epoch.iter <- 1
    optimParRS.collect <- list()
  } else {
    epoch.iter <- epoch.iter + 1
    optimParRS.collect <- list(model_output[["optimParRS"]])
  } 
  
  if (optim.sensitivity >1 | optim.sensitivity < -1) {stop("optim.sensitivity must be between -1 and 1")}
  assign("optim.sensitivity", optim.sensitivity, envir = .GlobalEnv)
  
  while (optim.loop & optim.iter <= epochN) {
    
    assign("epoch.iter", epoch.iter, envir = .GlobalEnv)
    assign("optim.iter", optim.iter, envir = .GlobalEnv)
    # run RS model with adapted 
    
    sysTimeRS <- system.time({
      model_output <- f.mmm(hyper_bound_global
                        ,iterRS = set_iter
                        ,set_cores = set_cores
                        ,out = out
      )})
    
    # get optimum parameters based on mode of top10% mape density
    optimParRS <- f.getOptimParRS(model_output, kurt.tuner = optim.sensitivity)
    
    if(optim.iter==1 & epoch.iter==1) {
      optimParRS[, epoch.optim := 0]
      epoch.optim.update <- optimParRS$optim.found*1
    } else {
      optimParRS[, epoch.optim:= epoch.optim.update]
      epoch.optim.update <- epoch.optim.update + optimParRS$optim.found*1
    }
    optimParRS[, epochN:= epoch.iter-1]
    assign("epoch.optim.update", epoch.optim.update, envir = .GlobalEnv)
    
    if(any(optimParRS$optim.found)) {
      
      # get optimised bounds
      param.optim <- optimParRS[optim.found == T, variable]
      param.noOptim <- optimParRS[optim.found == F, variable]
      
      cat("\n## Hyperbounds optimisation epoch", epoch.iter-1, "took", round(sysTimeRS[3]/3600,4),"hours to run",set_iter,"iterations...\n##"
          ,length(param.optim), "out of",nrow(optimParRS),"hyperparameters found optimum:" , param.optim, "\n##"
          ,"Updated bounds are...\n\n")
      print(optimParRS[, !c("kurt"), with = F])
      
      # update bounds
      hyperbound.local.auto <- lapply(hyperparameters, range)
      hyperbound.local.auto <- mapply(function(x, xName, low, up) {
        if(xName %in% param.optim) {
          x <- c(low, up)
          return(x)
        } else {return(x)}
      }, x=hyperbound.local.auto, xName = names(hyperbound.local.auto), low= optimParRS$low, up = optimParRS$up, SIMPLIFY = F)
      
      assign("hyperbound.local.auto", hyperbound.local.auto, envir = .GlobalEnv)
      
      lhsOut <- f.hypSamLHS(set_mediaVarName, set_iter = set_iter, hyper_bound_global, adstock)
      
      hyperparameters <- lhsOut$transLHS.list
      assign("hyperparameters", hyperparameters, envir = .GlobalEnv)
      
      optim.iter <- optim.iter + 1
      epoch.iter <- epoch.iter + 1
    } else {
      
      optimParRS[, epoch.optim:= epoch.optim.update]
      #optimParRS[, optim.found:= epoch.optim.update>0]
      cat("\n####################################\nAfter"
          , epoch.iter-1, "epoches, no further hyperparameter optimisation can be found. Final bounds are...\n\n")
      print(optimParRS[, !c("kurt"), with = F])
      cat("####################################\n")
      if (epoch.iter == 1) {lhsOut <- NULL}
      optim.loop <- F
      epoch.iter <- epoch.iter + 1
    }
    optimParRS.collect[[epoch.iter]] <- optimParRS
    fwrite(optimParRS, temp.csv.path)
    closeAllConnections()
  } # while loop end 
  
  if (optim.iter>epochN) {
    optimParRS[, epoch.optim:= epoch.optim.update]
    #optimParRS[, optim.found:= epoch.optim.update>0]
    cat("\n####################################\nThere are still local optimum found. Increase epochN to reach optimum. Current bounds are...\n\n")
    print(optimParRS[, !c("kurt"), with = F])
    cat("####################################\n")
  }
  
  model_output[["lhsOut"]] <- lhsOut
  model_output[["optimParRS"]] <- rbindlist(optimParRS.collect)
  return(model_output)
}

#####################################
#### Define result collection function

f.mmmCollect <- function(optimParRS) { # optimParRS = model_output$optimParRS
  
  getSpendSum <- dt_input[, lapply(.SD, sum), .SDcols=set_mediaSpendName]
  names(getSpendSum) <- set_mediaVarName
  getSpendSum <- suppressWarnings(melt.data.table(getSpendSum, measure.vars= set_mediaVarName, variable.name = "rn", value.name = "spend"))
  
  if (all(sapply(set_hyperBoundLocal, length)==1) & !is.null(set_hyperBoundLocal)) {
    model_output <- f.mmm(set_hyperBoundLocal, out = T)
    model_output$resultCollect$xDecompAgg <- model_output$resultCollect$xDecompAgg[getSpendSum, on = "rn", spend:=i.spend]
    model_output$resultCollect$xDecompAgg[, roi := xDecompAgg / spend]
    assign("model_output", model_output, envir = .GlobalEnv)

  } else {
    optimParRS.last <- optimParRS[epochN==max(epochN)]
    bestParRS <- lapply(optimParRS.last$mode, function(x) x)
    names(bestParRS) <- optimParRS.last$variable
    set_hyperBoundLocal <- bestParRS
    model_output <- f.mmm(bestParRS, out = T)
    model_output$resultCollect$xDecompAgg <- model_output$resultCollect$xDecompAgg[getSpendSum, on = "rn", spend:=i.spend]
    model_output$resultCollect$xDecompAgg[, roi := xDecompAgg / spend]
  }

  return(model_output)
}

