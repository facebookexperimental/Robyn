# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#############################################################################################
####################    Facebook MMM Open Source 'Robyn' Beta - V21.0  ######################
####################                    2021-03-03                     ######################
#############################################################################################

################################################################
#### set locale for non English R
# Sys.setlocale("LC_TIME", "English")

################################################################
#### load libraries
## R version 4.0.3 (2020-10-10) ## Update to R version 4.0.3 to avoid potential errors
## RStudio version 1.2.1335
rm(list=ls()); gc()

## Please make sure to install all libraries before rurnning the scripts
library(data.table) 
library(stringr) 
library(lubridate) 
library(doParallel) 
library(foreach) 
library(glmnet) 
library(car) 
library(StanHeaders)
library(prophet)
library(ggplot2)
library(gridExtra)
library(grid)
library(ggpubr)
library(see)
library(PerformanceAnalytics)
library(nloptr)
library(minpack.lm)
library(rPref)
library(reticulate)
library(rstudioapi)

## please see https://rstudio.github.io/reticulate/index.html for info on installing reticulate
# conda_create("r-reticulate") # must run this line once
# conda_install("r-reticulate", "nevergrad", pip=TRUE)  #  must install nevergrad in conda before running Robyn
# use_python("/Users/gufengzhou/Library/r-miniconda/envs/r-reticulate/bin/python3.6") # in case nevergrad still can't be imported after installation, please locate your python file and run this line
use_condaenv("r-reticulate") 

################################################################
#### load data & scripts
script_path <- str_sub(rstudioapi::getActiveDocumentContext()$path, start = 1, end = max(unlist(str_locate_all(rstudioapi::getActiveDocumentContext()$path, "/"))))
dt_input <- fread(paste0(script_path,'de_simulated_data.csv')) # input time series should be daily, weekly or monthly
dt_holidays <- fread(paste0(script_path,'holidays.csv')) # when using own holidays, please keep the header c("ds", "holiday", "country", "year")

source(paste(script_path, "fb_robyn.func.R", sep=""))
source(paste(script_path, "fb_robyn.optm.R", sep=""))

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

## set model core features
adstock <- "geometric" # geometric or weibull. weibull is more flexible, yet has one more parameter and thus takes longer
set_iter <- 500  # number of allowed iterations per trial. 500 is recommended

set_hyperOptimAlgo <- "DiscreteOnePlusOne" # selected algorithm for Nevergrad, the gradient-free optimisation library https://facebookresearch.github.io/nevergrad/index.html
set_trial <- 40 # number of allowed iterations per trial. 40 is recommended without calibration, 100 with calibration.
## Time estimation: with geometric adstock, 500 iterations * 40 trials and 6 cores, it takes less than 1 hour. Weibull takes at least twice as much time.

## helper plots: set plot to TRUE for transformation examples
f.plotAdstockCurves(F) # adstock transformation example plot, helping you understand geometric/theta and weibull/shape/scale transformation
f.plotResponseCurves(F) # s-curve transformation example plot, helping you understand hill/alpha/gamma transformation

################################################################
#### tune channel hyperparameters bounds

#### Guidance to set hypereparameter bounds #### 

## 1. get correct hyperparameter names: 
local_name <- f.getHyperNames(); local_name # names in set_hyperBoundLocal must equal names in local_name, case sensitive

## 2. get guidance for setting hyperparameter bounds:
# For geometric adstock, use theta, alpha & gamma. For weibull adstock, use shape, scale, alpha, gamma
# theta: In geometric adstock, theta is decay rate. guideline for usual media genre: TV c(0.3, 0.8), OOH/Print/Radio c(0.1, 0.4), digital c(0, 0.3)
# shape: In weibull adstock, shape controls the decay shape. Recommended c(0.0001, 2). The larger, the more S-shape. The smaller, the more L-shape
# scale: In weibull adstock, scale controls the decay inflexion point. Very conservative recommended bounce c(0, 0.1), becausee scale can increase adstocking half-life greaetly
# alpha: In s-curve transformation with hill function, alpha controls the shape between exponential and s-shape. Recommended c(0.5, 3). The larger the alpha, the more S-shape. The smaller, the more C-shape
# gamma: In s-curve transformation with hill function, gamma controls the inflexion point. Recommended bounce c(0.3, 1). The larger the gamma, the later the inflection point in the response curve 
   
## 3. set each hyperparameter bounds. They either contains two values e.g. c(0, 0.5), or only one value (in which case you've "fixed" that hyperparameter)
set_hyperBoundLocal <- list(
  facebook_I_alphas = c(0.5, 3) # example bounds for alpha
 ,facebook_I_gammas = c(0.3, 1) # example bounds for gamma
 ,facebook_I_thetas = c(0, 0.3) # example bounds for theta
 #,facebook_I_shapes = c(0.0001, 2) # example bounds for shape
 #,facebook_I_scales = c(0, 0.1) # example bounds for scale
  
  ,ooh_S_alphas = c(0.5, 3)
  ,ooh_S_gammas = c(0.3, 1)
  ,ooh_S_thetas = c(0.1, 0.4) 
 #,ooh_S_shapes = c(0.0001, 2)
 #,ooh_S_scales = c(0, 0.1)
  
  ,print_S_alphas = c(0.5, 3) 
  ,print_S_gammas = c(0.3, 1)
 ,print_S_thetas = c(0.1, 0.4)
 #,print_S_shapes = c(0.0001, 2)
 #,print_S_scales = c(0, 0.1)
  
  ,tv_S_alphas = c(0.5, 3) 
  ,tv_S_gammas = c(0.3, 1)
  ,tv_S_thetas = c(0.3, 0.8)
 #,tv_S_shapes = c(0.0001, 2)
 #,tv_S_scales= c(0, 0.1)
  
  ,search_clicks_P_alphas = c(0.5, 3)  
  ,search_clicks_P_gammas = c(0.3, 1)
  ,search_clicks_P_thetas = c(0, 0.3)
 #,search_clicks_P_shapes = c(0.0001, 2)
 #,search_clicks_P_scales = c(0, 0.1)
  
)

################################################################
#### define ground truth (e.g. Geo test, FB Lift test, MTA etc.)

activate_calibration <- F # Switch to TRUE to calibrate model.
# set_lift <- data.table(channel = c("facebook_I",  "tv_S", "facebook_I"),
#                        liftStartDate = as.Date(c("2018-05-01", "2017-11-27", "2018-07-01")),
#                        liftEndDate = as.Date(c("2018-06-10", "2017-12-03", "2018-07-20")),
#                        liftAbs = c(400000, 300000, 200000))

################################################################
#### Prepare input data

dt_mod <- f.inputWrangling() 

################################################################
#### Run models

model_output_collect <- f.robyn(set_hyperBoundLocal
                                ,optimizer_name = set_hyperOptimAlgo
                                ,set_trial = set_trial
                                ,set_cores = set_cores
                                ,plot_folder = "~/Documents/GitHub/plots") # please set your folder path to save plots. It ends without "/".

################################################################
#### Budget Allocator - Beta

## Budget allocator result requires further validation. Please use this result with caution.
## Please don't interpret budget allocation result if there's no satisfying MMM result

model_output_collect$allSolutions
optim_result <- f.budgetAllocator(modID = "3_11_5" # input one of the model IDs in model_output_collect$allSolutions to get optimisation result
                                  ,scenario = "max_historical_response" # c(max_historical_response, max_response_expected_spend)
                                  #,expected_spend = 100000 # specify future spend volume. only applies when scenario = "max_response_expected_spend"
                                  #,expected_spend_days = 90 # specify period for the future spend volumne in days. only applies when scenario = "max_response_expected_spend"
                                  ,channel_constr_low = c(0.7, 0.75, 0.60, 0.8, 0.65) # must be between 0.01-1 and has same length and order as set_mediaVarName
                                  ,channel_constr_up = c(1.2, 1.5, 1.5, 2, 1.5) # not recommended to 'exaggerate' upper bounds. 1.5 means channel budget can increase to 150% of current level
)

