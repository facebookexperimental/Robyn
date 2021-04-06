---
id: quick-start
title: Quick Start
---

import useBaseUrl from '@docusaurus/useBaseUrl';


**Once you have loaded all scripts, we will focus on the ‘fb_robyn.exec.R’ one to quickly start testing the code with the included simulated data and understand how to use it. Below you will find the steps to follow:**

## Load packages

First, install and load all packages before running the code. You will utilize several open source packages to run it. You will find several packages related to working with data tables, loops, parallel computing and plotting results. However, the main package for the core regression process is [‘glmnet’](https://cran.r-project.org/web/packages/glmnet/index.html) from which the ridge regression will execute . Another important package is [‘reticulate’](https://rstudio.github.io/reticulate/) which provides a comprehensive set of tools for interoperability between Python and R and will be key to be able to work with [Nevergrad’s](https://facebookresearch.github.io/nevergrad/) algorithms.


#### Please make sure to install all libraries before running the scripts

```
library(glmnet)
library(reticulate)
...
...
```

## Create, install and use conda environment for Nevergrad
Once you have installed and loaded all packages you will need to execute the following commands in order to create, install and use conda environments on reticulate. This is required to be able to use Nevergrad algorithms which use Python:

```
conda_create("r-reticulate") #must run this line once only
conda_install("r-reticulate", "nevergrad", pip=TRUE) #must install nevergrad in conda before running Robyn
use_condaenv("r-reticulate")
```


## Load data

First you will load the included simulated data and create the outcome variable.
As in any MMM, this is a dataframe with a minimum set of columns ds and y, containing the date and numeric value respectively. You may also want to add explanatory variables to account for different marketing channels and their investment, impressions or any other metric to determine the size and impact of marketing campaigns.
Please have in mind that this automated file reading solution requires that you are using RStudio and that it will set your working directory as the source file location in Rstudio:


```
#### load data & scripts

script_path <- str_sub(rstudioapi::getActiveDocumentContext()$path, start = 1, end = max(unlist(str_locate_all(rstudioapi::getActiveDocumentContext()$path, "/"))))
dt_input <- fread(paste0(script_path,'de_simulated_data.csv')) # input time series should be daily, weekly or monthly
dt_holidays <- fread(paste0(script_path,'holidays.csv')) # when using own holidays, please keep the header c("ds", "holiday", "country", "year")

source(paste(script_path, "fb_robyn.func.R", sep=""))
source(paste(script_path, "fb_robyn.optm.R", sep=""))
```

## Set model input variables

The next step is to define the variables you will be working with from the previously uploaded data. There are different types of variables, the main three ones are: dependent (set_depVarName), date (set_dateVarName) and media variable names (set_mediaVarName).

```
#### set model input variables
set_dateVarName <- c("DATE") # date format must be "2020-01-01"
set_depVarName <- c("revenue") # there should be only one dependent variable
set_mediaVarName <- c("tv_S"	,"ooh_S",	"print_S"	,"facebook_I"	,"search_clicks_P")
```

The following set of variables are related to [Prophet](https://facebook.github.io/prophet/), the open source procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality. "trend","season", "weekday", "holiday" are provided and case-sensitive. We recommend at least to keep trend and holidays if you decide to use it.
```
set_prophet <- c("trend", "season", "holiday")
```

Moreover, you will have to define which base variables (set_baseVarName) typically competitors, price, promotion, temperature, unemployment rate, etc. you own to use. In the example simulated data we use competitor_sales_B as one of the baseline variables:

```
set_baseVarName <- c("competitor_sales_B")
```

Finally, you will find sign controls for variables, these will control for constrained variables that theoretically have to be greater than zero (positive), lower than zero (negative), or can take just any coefficient values (default).You will see there are media and base variables sign control so you will have to define them on separate variables:


```
set_mediaVarSign <- c("positive", "positive", "positive", "positive", "positive")

set_prophetVarSign <- c("default","default", "default")

set_baseVarSign <- c("negative")
```

## Set cores for parallel computing

Next we will define the amount of cores to allocate for computing to the overall process. Please have in mind to always leave one or two cores out of the total number of cores your machine has to prevent your OS from crashing. Use detectCores() to find out the number of cores in your machine.

```
## set cores for parallel computing
registerDoSEQ(); detectCores()
set_cores <- 6 # Use detectCores() to find out cores
```

## Set model training size
The next variable to set is the training size. If we set set_modTrainSize to 0.74 it means we will leave 74% of the data for training and 26% for validation. Use f.plotTrainSize to get a split estimation. Please balance between a higher Bhattacharyya coefficient and a sufficient training size. If you use the simulated data you may just leave it as it is (0.74)

```
## set training size
f.plotTrainSize(F)
set_modTrainSize <- 0.74
```
## Set model core features
The following step is crucial, this is where you will define if you will be using weibull or geometric adstock functions. Weibull is more flexible, yet has one more parameter and thus takes longer. (Please refer to the variables transformation section within this documentation for more information).
You will also need to define the number of iterations per trial for the algorithm to find optimal hyperparameter values. 500 is recommended, but just to test it on provided simulated data and reduce total computing time you may reduce this number.

After that, you will find set_hyperOptimAlgo <- "DiscreteOnePlusOne" which is the selected algorithm for [Nevergrad](https://facebookresearch.github.io/nevergrad/index.html), the gradient-free optimization library. There is no need to change the algorithm, however there are several to choose from if you wanted to.
Finally, under ‘set_trial’ you will have to define the number of trials. 40 is recommended without calibration, 100 with calibration. If set_iter <- 500 and set_trial <- 40, this means that we will have 40 different initialization trials, each of them with 500 iterations.

```
#### set model core features
adstock <- "geometric" # geometric or weibull.
set_iter <- 500
set_hyperOptimAlgo <- "DiscreteOnePlusOne"
set_trial <- 40
```


## Set hyperparameters bounds

This is an optional step, we recommend you leave it as it is at the beginning, as there is absolutely no need to change it. You may edit bounds in case you already found optimal ranges for parameters after obtaining first results. For geometric adstock, use theta, alpha and gamma. For weibull adstock, use shape, scale, alpha and gamma.

```
#### set hyperparameters

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

```

## Define ground-truth calibration

The last step in variable definition is to add ground-truth data such as incremental studies, like conversion lift data from Facebook, geo tests or Multi-touch attribution. You will need to first define activate_calibration <- T to include calibration in your model. Consequently, you will need to define which channels you want to define certain incremental values for, as well as, the start date, end date and incremental absolute values (liftAbs) from the studies.

```
#### define ground truth
activate_calibration <- F # Switch to TRUE to calibrate model.
# set_lift <- data.table(channel = c("facebook_I",  "tv_S", "facebook_I"),
#                        liftStartDate = as.Date(c("2018-05-01", "2017-11-27", "2018-07-01")),
#                        liftEndDate = as.Date(c("2018-06-10", "2017-12-03", "2018-07-20")),
#                        liftAbs = c(400000, 300000, 200000))
```

## Prepare the input data and run the models

Once you have defined all the variables from previous steps, you will need to first prepare the input data within the ‘dt_mod’ object. Finally, you will have to execute the ‘f.robyn’ function in order to run the model.  Please set your preferred folder path (If the default "~/Documents/GitHub/plots" does not work out well for you) to save plots. Please notice the path has to end without a "/".

```
#### Prepare input data
dt_mod <- f.inputWrangling()

#### Run models
model_output_collect <- f.robyn(set_hyperBoundLocal
                                ,optimizer_name = set_hyperOptimAlgo
                                ,set_trial = set_trial
                                ,set_cores = set_cores
                                ,plot_folder = "~/Documents/GitHub/plots")
```

## Access plotted results

Once all trials and iterations are finished, the model will proceed to plot different charts that will help you assess best models based on NRMSE and a media variables decomposition quality score for contribution of marketing channels. You may find all model plots like below example within the plot_folder you have set on the ‘run models’ step.

<img alt="Model results" src={useBaseUrl('/img/ModelResults1.png')} />
