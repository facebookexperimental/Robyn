# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#############################################################################################
####################    Facebook MMM Open Source 'Robyn' Beta - V20.0  ######################
####################                    2020-11-30                     ######################
#############################################################################################
################################################################
#### set locale for non English R
#Sys.setlocale("LC_TIME", "English")

################################################################
#### load libraries
## R version 3.6.0 (2019-04-26)
## RStudio version 1.2.1335
rm(list=ls()); gc()

## Please make sure to install all libraries before rurnning the scripts
library(data.table) # version 1.12.2
library(stringr) # version 1.4.0
library(lubridate) # version 1.7.4
library(doParallel) # version 1.0.15
library(doSNOW) # version 1.0.18
library(foreach) # version 1.4.8
library(glmnet) # version 2.0.18
library(lhs) # version 1.0.1
library(car) # version 3.0.3
library(StanHeaders) # version 2.21.0
library(prophet) # version 0.5
library(ggplot2) # version 3.3.0
library(gridExtra) # version 2.3
library(see) # version 0.5.0
library(PerformanceAnalytics) # version 2.0.4
library(nloptr) # version 1.2.1
library(minpack.lm) # version 1.2
library(reticulate)

################################################################
#### load data & scripts
script_path <- str_sub(rstudioapi::getActiveDocumentContext()$path, start = 1, end = max(unlist(str_locate_all(rstudioapi::getActiveDocumentContext()$path, "/"))))
# script_path <- ""
dt_input <- fread(paste0(script_path,'de_simulated_data.csv')) # input time series should be daily, weekly or monthly
dt_holidays <- fread(paste0(script_path,'holidays.csv')) # when using own holidays, please keep the header c("ds", "holiday", "country", "year")

source(paste(script_path, "fb_nextgen_mmm_v20.0.func.R", sep=""))
source(paste(script_path, "fb_nextgen_mmm_v20.0.plot.R", sep=""))
source(paste(script_path, "fb_nextgen_mmm_v20.0.optm.R", sep=""))

################################################################
#### set model input variables

set_country <- "DE" # only one country allowed once. Including national holidays for 59 countries, whose list can be found on our githut guide 
set_dateVarName <- c("DATE") # date format must be "2020-01-01"
set_depVarName <- c("revenue") # there should be only one dependent variable
set_depVarType <- "revenue" # "revenue" or "conversion" are allowed

activate_prophet <- T # Turn on or off the Prophet feature
set_prophet <- c("trend", "season", "holiday") # "trend","season", "weekday", "holiday" are provided and case-sensitive. Recommended to at least keep Trend & Holidays
set_prophetVarSign <- c("default","default", "default") # c("default", "positive", and "negative"). Recommend as default. Must be same length as set_prophet

activate_baseline <- T
set_baseVarName <- c("competitor_sales_B") # typically competitors, price & promotion, temperature,  unemployment rate etc
set_baseVarSign <- c("negative") # c("default", "positive", and "negative"), control the signs of coefficients for baseline variables

set_mediaVarName <- c("tv_S"	,"ooh_S",	"print_S"	,"facebook_I"	,"search_clicks_P") # c("tv_S"	,"ooh_S",	"print_S"	,"facebook_I", "facebook_S"	,"search_clicks_P"	,"search_S") we recommend to use media pressure metrics like impressions, GRP etc for the model. If not applicable, use spend instead
set_mediaVarSign <- c("positive", "positive", "positive", "positive", "positive") # c("default", "positive", and "negative"), control the signs of coefficients for media variables
set_mediaSpendName <- c("tv_S"	,"ooh_S",	"print_S"	,"facebook_S"	,"search_S") # spends must have same order and same length as set_mediaVarName

set_factorVarName <- c() # please specify which variable above should be factor, otherwise leave empty c()

################################################################
#### set global model parameters

## set cores for parallel computing
registerDoSEQ(); detectCores()
set_cores <- 6 # I am using 6 cores from 8 on my local machine. Use detectCores() to find out cores

## set training size
f.plotTrainSize(F) # insert TRUE to plot training size guidance. Please balance between higher Bhattacharyya coefficient and sufficient training size
set_modTrainSize <- 0.74 # 0.74 means taking 74% of data to train and 30% to test the model. Use f.plotTrainSize to get split estimation

## set model core features/
adstock <- "geometric" # geometric or weibull . weibull is more flexible, yet has one more parameter and thus takes longer
set_iter <- 50000  #50000 # We recommend to run at least 50k iteration at the beginning, when hyperparameter bounds are not optimised

# no need to change
f.plotAdstockCurves(F) # adstock transformation example plot, helping you understand geometric/theta and weibull/shape/scale transformation
f.plotResponseCurves(F) # s-curve transformation example plot, helping you understand hill/alpha/gamma transformation
set_hyperBoundGlobal <- list(thetas = c(0, 0.3) # geometric decay rate
                          ,shapes = c(0.0001, 2) # weibull parameter that controls the decay shape between exponential and s-shape. The larger the shape, the more S-shape. The smaller, the more L-shape
                          ,scales = c(0, 0.05) # weibull parameter that controls the position of inflection point. Be very careful with scale, because moving inflexion point has strong effect to adstock transformation
                          ,alphas = c(0.5, 3) # hill function parameter that controls the shape between exponential and s-shape. The larger the alpha, the more S-shape. The smaller, the more L-shape
                          ,gammas = c(0.3, 1) # hill functionn pararmeter that controls the scale of trarnsforrmation. The larger the gamma, the later the inflexion point in the response curve 
                          ,lambdas = c(0, 1)) # regularised regression parameter
global_name <- names(set_hyperBoundGlobal)

################################################################
#### tune channel hyperparameters bounds

activate_hyperBoundLocalTuning <- F # change setChannelBounds = T when setting bounds for each media individually
local_name <- f.getHyperNames(); local_name # get hyperparameter names for each channel. channel bound names must be identical as in local_name

## channel bounds have to stay within set_hyperBoundGlobal as specified above. Unhide set_hyperBoundLocal below to set channel level bounds
## each bound can be either a range (e.g c(0.1,0.3)) or one fixed value

set_hyperBoundLocal <- list(
  facebook_I_alphas = c(2, 3) # example bounds for digital channels: the larger alpha, the more S-shape for response curve
  ,facebook_I_gammas = c(0.0001, 0.2) # example bounds for digital channels: the smaller gamma, the earlier inflexion point occurs
  ,facebook_I_thetas = c(0, 0.2) # example bounds for digital channels: the smaller theta for geometric adstock, the lower the decay/half-life

  ,ooh_S_alphas = c(0.0001, 1) # example bounds for traditional channels: the smaller alpha, the more L-shape for response curve
  ,ooh_S_gammas = c(0.0001, 0.5) # example bounds for traditional channels: the larger gamma, the later inflexion point occurs
  ,ooh_S_thetas = c(0.3, 0.5) # example bounds for digital channels: the larger theta for geometric adstock, the higher the decay/half-life

  ,print_S_alphas = c(0.0001, 1)
  ,print_S_gammas = c(0.0001, 0.5)
  ,print_S_thetas = c(0.3, 0.5)

  ,tv_S_alphas = c(0.0001, 1)
  ,tv_S_gammas = c(0.0001, 0.5)
  ,tv_S_thetas = c(0.3, 0.5)

  ,search_clicks_P_alphas = c(2, 3)
  ,search_clicks_P_gammas = c(0.0001, 0.2)
  ,search_clicks_P_thetas = c(0, 0.1)

)

################################################################
#### define experimental results

activate_calibration <- F # Switch to TRUE to calibrate model. This takes longer as extra validation is required
# set_lift <- data.table(channel = c("facebook_I",  "tv_S", "facebook_I"),
#                        liftStartDate = as.Date(c("2018-05-01", "2017-11-27", "2018-07-01")),
#                        liftEndDate = as.Date(c("2018-06-10", "2017-12-03", "2018-07-20")),
#                        liftAbs = c(70000000, 5000000, 50000000))

################################################################
#### Prepare input data

dt_mod <- f.inputWrangling() 

################################################################
#### Run models

model_output <- f.mmmRobyn(set_hyperBoundGlobal
                          ,set_iter = set_iter
                          ,set_cores = set_cores
                          ,epochN = Inf # set to Inf to auto-optimise until no optimum found
                          ,optim.sensitivity = 0 # must be from -1 to 1. Higher sensitivity means finding optimum easier
                          ,temp.csv.path = './mmm.tempout.csv' # output optimisation result for each epoch. Use getwd() to find path
                          )

best_model <- f.mmmCollect(model_output$optimParRS)
# best_model <- f.mmmCollect(set_hyperBoundLocal)

################################################################
#### Plot section

## insert TRUE into plot functions to plot. Use 'channelPlot' to select max. 3 channels per plot
f.plotSpendModel(F)
f.plotHyperSamp(F, channelPlot = c("tv_S", "ooh_S", "facebook_I")) # plot latin hypercube hyperparameter sampling balance. Max. 3 channels per plot
f.plotTrendSeason(F) # plot prophet trend, season and holiday decomposition
bestAdstock <- f.plotMediaTransform(F, channelPlot = c("tv_S", "ooh_S", "facebook_I")) # 3 plots of best model media transformation: adstock decay rate, adstock effect & response curve. Max. 3 channels per plot
f.plotBestDecomp(F) # 3 plots of best model decomposition: sales decomp, actual vs fitted over time, & sales decomp area plot
f.plotMAPEConverge(F) # plot RS MAPE convergence, only for random search
f.plotBestModDiagnostic(F) # plot best model diagnostics: residual vs fitted, QQ plot and residual vs. actual
f.plotChannelROI(F)
f.plotHypConverge(F, channelPlot = c("tv_S", "ooh_S", "facebook_I")) # plot hyperparameter vs MAPE convergence. Max. 3 channels per plot
boundOptim <- f.plotHyperBoundOptim(F, channelPlot = c("tv_S", "ooh_S", "facebook_I"), model_output, kurt.tuner = optim.sensitivity)  # improved hyperparameter plot to better visualise trends in each hyperparameter


################################################################
#### Optimiser - Beta

## Optimiser requires further validation. Please use this result with caution.
## Please don't interpret optimiser result with intermediate MMM output.
## Optimiser result is only interpretable when MMM result is finalised/ hyperparameters are fixed. 

optim_result <- f.optimiser(scenario = "max_historical_response" # c(max_historical_response, max_response_expected_spend)
                            #,expected_spend = 100000 # specify future spend volume. only applies when scenario = "max_response_expected_spend"
                            #,expected_spend_days = 90 # specify period for the future spend volumne in days. only applies when scenario = "max_response_expected_spend"
                            ,channel_constr_low = c(0.7, 0.75, 0.60, 0.5, 0.65) # must be between 0.01-1 and has same length and order as set_mediaVarName
                            ,channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5) # not recommended to 'exaggerate' upper bounds. 1.5 means channel budget can increase to 150% of current level
                            ) 

print(optim_result$dt_optimOut)
f.plotOptimiser(F) # 3 plots of optimiser result: budget re-allocation, ROI comparison & response comparison 
