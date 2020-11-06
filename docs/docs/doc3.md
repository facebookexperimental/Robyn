---
id: doc3
title: Quick Start
---

**Once you have loaded all scripts, we will focus on the ‘.exec.R’ one to quickly start testing the code and understanding how to use it. Below you will find the steps to follow:**  


## Load packages
You will utilize several open source packages to run this code, please install and load all libraries before running it. You will find several packages related to working with data tables, loops, parallel computing and plotting results, however the package for the core regression process is library(glmnet) from which the ridge regression will execute. 

#### Please make sure to install all libraries before running the scripts
``` 
library(glmnet)
...
...
```


## Load data
First you will load the data and create the outcome variable. As in any MMM, this is a dataframe with a minimum set of columns ds and y, containing the date and numeric value respectively. You may also want to add regressors variables to account for different marketing channels and their investment, impressions or any other metric to determine the size and impact of marketing campaigns. 
Please have in mind that this automated file reading solution requires that you are using RStudio and that it will set your working directory as the source file location in Rstudio.

``` 
#### load data
script_path <- str_sub(rstudioapi::getActiveDocumentContext()$path, start = 1, end = max(unlist(str_locate_all(rstudioapi::getActiveDocumentContext()$path, "/"))))
dt_input <- fread(paste0(script_path,'de_simulated_data.csv'))
holidays <- fread(paste0(script_path,'generated_holidays.csv')
``` 

## Set global parameters
The next step is to define the variables you will be working with from the previously uploaded data. There are different types of variables, as mentioned above, the main three ones are dependent (set_depVarName), date (set_dateVarName) and media volume (set_mediaVarName). 

``` 
#### define variables
set_dateVarName <- c("DATE") # date must be format "2020-01-01"
set_depVarName <- c("revenue") # there should be only one dependent variable
set_mediaVarName <- c("tv_S", "facebook_I" ) # c("revenue", "tv_S", "ooh_S", "print_S", "facebook_I"	, "search_clicks_P", "search_imps_P", "search_S", "competitor_sales_B") we recommend to use media pressure metrics like impressions, GRP etc for the model. If not applicable, use spend instead
``` 

Moreover, You will have to define which base variables (set_baseVarName) provided by the code or that you own to use, we recommend at least to keep the trend and holidays in the model.
``` 
set_baseVarName <- c("TREND","HOLIDAYS","SEASONAL", "competitor_sales_B") 
##### "TREND", "HOLIDAYS","SEASONAL", "WEEKDAY", "HOURLY" are provided by the code. 
``` 

Finally, you will find two variables for sign control, these will control for constrained variables that theoretically have to be greater than zero (positive), lower than zero (negative), or can take just any coefficient values (default).You will see there are media and base variables sign control so you will have to define them on separate variables:
``` 
set_mediaVarSign <- c("positive", "positive") # c("default", "positive", and "negative"), control the signs of coefficients for media variables
set_baseVarSign <- c("default", "default", "default", "negative") # c("default", "positive", and "negative"), control the signs of coefficients for base variables
``` 


## Set cores for parallel computing
Next we will define the amount of cores to allocate to the overall process. Please bear in mind to always leave one or two cores out of the total number of cores your machine has to prevent your OS from crashing.
``` 
#### set cores for parallel computing
registerDoSEQ(); detectCores()
setCores <- 6
``` 

## Set model core features
The following step is crucial, this is where you will define if you will be using weibull or geometric adstock functions (Please refer to the variables transformation section within this documentation). You will also need to define the number of iterations for the algorithm to loop and find optimal hyperparameter values.
``` 
#### set model core features
adstock <- "geometric" # geometric or weibull
iterN <- 1000 # "rs" iteration is theoretically unlimited.
``` 

## Set hyperparameters bounds
This is an optional step as there is absolutely no need to change it. You may edit bounds in case you already found optimal ranges for parameters after several iterations. We recommend you leave it as it is at the beginning.
``` 
#### set hyperparameters
no need to change
hypBound <- list(thetas = c(0, 0.9999) ,shapes = c(0, 5) ,scales = c(0.0001, 0.9999), 
                 alphas = c(0, 5) ,gammas = c(0.0001, 0.9999) ,lambdas = c(0, 1))
``` 

## Set model train and test size
On this step you will define the percentage of your data you will be saving to test the model once it has been trained and validated. We recommend assigning 80% for training purposes.
``` 
set_mod_train_size <- 0.8 # 0.8 means taking 80% of data to train and 20% to test the model
``` 

## Define experimental results and calibration
The last step in variable definition is to add incremental studies data in case you have information available, such as conversion lift data for Facebook. You will need to first define calibrateLift <- T to include calibration in your model. Consequently, you will need to define which channels you want to define certain incremental values for as well as, start, end and incremental absolute values (liftAbs) from the studies. 
``` 
calibrateLift <- F
set_lift <- data.table(channel = c("facebook_I",  "tv_S", "facebook_I"),
                       liftStartDate = as.Date(c("2018-05-01", "2017-11-27", "2018-07-01")),
                       liftEndDate = as.Date(c("2018-06-10", "2017-12-03", "2018-07-20")),
                       liftAbs = c(70000000, 5000000, 50000000))
``` 


## Loading scripts and running the model

Once you have defined all the variables from previous steps, you will need to finally execute the ‘.func.R’ and ’.plot.R’ scripts in order to run the model. Therefore you will need to load the scripts first, run the models and print results as per below: 
``` 
#### load scripts
source(paste(script_path, "fb_nextgen_mmm_v19.func.R", sep=""))
source(paste(script_path, "fb_nextgen_mmm_v19.bayes.R", sep=""))
source(paste(script_path, "fb_nextgen_mmm_v19.plot.R", sep=""))
``` 
``` 
#### Run model scripts
if (hyperparamOptim == "rs") {
  sysTimeRS <- system.time({
    resultRS <- f.mmm(hyperparameters,
                      iterRS = iterN,
                      hyperparamOptim = "rs",
                      setCores = setCores
    )})
  
  #print(head(resultRS$resultCollect$resultHypParam, 15))
  bestParRS <- f.getBestParRS(resultRS, calibrateLift)
  best.resultRS <- f.mmm(bestParRS, hyperparamOptim = "rs")
  
} }
registerDoSEQ(); getDoParWorkers()
``` 
## Plotting results

Once all iterations are finished you will proceed to plot different charts that will help you assess the models accuracy, business contribution for marketing channels and baseline variables.
``` 
#### insert TRUE into plot functions to plot
f.plotHyperSamp(F) # plot latin hypercube hyperparameter sampling balance
f.plotTrendSeason(F) # plot prophet trend, season and holiday decomposition
f.plotBestMod(T) # plot best model with 5 plots: media adstocking, sales decomp, actual vs fitted over time, sales decomp area plot & channel response curve
f.plotMAPE.RS(F) # plot RS MAPE convergence, only for random search
f.plotBestModResid(F) # plot best model diagnostics: residual vs fitted, QQ plot and residual vs. actual
f.plotHypConverge(F) # plot hyperparameter vs MAPE convergence
``` 