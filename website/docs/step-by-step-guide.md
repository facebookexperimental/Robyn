---
id: step-by-step-guide
title: Step-by-step guide
---

import useBaseUrl from '@docusaurus/useBaseUrl';

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

Once you have defined the input files to work with and loaded to the source all the functions needed to run the code. You will need to define and set the input variables.


### set_country

The first variable to declare is the country. We recommend using only one country especially if you are planning to leverage prophet for trend and seasonality which automatically pulls holidays for the country you have selected and simplifies the process. Under simulated data we are using "DE" as example country.

```
set_country <- "DE" # only one country allowed. Used in prophet holidays
```

### set_dateVarName

For date variables you must have in mind that the DATE column in your dataset
has to be in "yyyy-mm-dd " format. ie. "2020-01-01"

```
set_dateVarName <- c("DATE") # date must be format "2020-01-01"
```

### set_depVarName and set_depVarType

Setting the dependent variable is basically the outcome you are trying to measure. We only accept one dependent variable under set_depVarName. This variable can take the form of revenue (Sales or profit in monetary values) or conversion (Number of transactions, units sold) which you will indicate when defining the set_depVarType variable.

```
set_depVarName <- c("revenue") # there should be only one dependent variable
set_depVarType <- "revenue" # "revenue" or "conversion" are allowed

```

### Set Prophet variables

#### activate_prophet

First you will need to indicate the model if you would like to turn on or off the [Prophet](https://facebook.github.io/prophet/) feature in the code to be used for seasonality, trend and holidays. T (True) means it is activated and F (False) deactivated. Prophet Implements a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.


```
activate_prophet <- T
```

#### set_prophet

The next step is to select which of the provided outcomes of Prophet you will use in the model. It is recommended to at least keep trend and holidays. Please have in mind that "trend","season", "weekday", "holiday" are provided and case-sensitive. For more detail on what each selection will do, look into the [CRAN documentation of the Prophet package](https://cran.r-project.org/web/packages/prophet/prophet.pdf) for the functions - holidays, yearly.seasonality, weekly.seasonality, daily.seasonality. Best practices would be to include at least “trend” and “holiday”.

```
set_prophet <- c("trend", "season", "holiday")
```

#### set_prophetVarSign

You may define the variable sign control for prophet variables to be "default", "positive", or "negative". If you are expecting coefficients for prophet variables such as "trend", "season", "holiday" to be default (either positive or negative), positive or negative. We recommend using default to give prophet the chance to detect either positive or negative overall effects in the response variable. There are cases where you may want to control for the sign. For example, Let´s imagine you already know that general sales growth trend is decreasing in time due to the reality of the business and other factors out of your control such as economic crises. Therefore you may want to indicate the algorithm to find only a negative relationship between trend and sales to reflect this scenario.
Please remember the object declared must be same length as set_prophet’s.

```
set_prophetVarSign <- c("default","default", "default")
```

### Set Baseline variables

The following step is to set the baseline variables which typically are things like competitors, pricing, promotions, temperature, unemployment rate and any other variable that is not media exposure but has a strong relationship with sales outcomes. You will need to indicate the model if you would like to turn on or off baseline variables in the code. T (True) means it is activated and F (False) deactivated. In most cases, there should be some non-media variables that will have an impact on your dependent variable so creating these baseline variables is important.

```
activate_baseline <- T
```

You may then define the different baseline variables you would like to consider. These should be the names of the columns in your data file.

```
set_baseVarName <- c("promotions", "price changes", "competitors sales")
```

You may apply sign control for baseline variables to be "default", "positive", or "negative". If you are expecting coefficients for baseline variables such as "promotions", "price changes", "competitors sales" to be default (either positive or negative), positive or negative depending on its expected relationship with the dependent variable. For example, rainy weather may have a positive or negative impact in sales depending on the business. Please remember the object declared must be same length as set_baseVarName’s

```
set_baseVarSign <- c("negative",’default’,’negative’) #“positive” is the remaining option
```

### set_mediaVarName and set_mediaSpendName

There is one key restriction to have in mind here, you must have spend variables declared for every media channel you would like to measure. So they have to be in the same order and same length as set_mediaVarName variables. If the data is available to include both an impressions/GRPs/etc (non-spend variable) in addition to the spend variable use the non-spend variable in set_mediaVarName and ensure that the variable name does not use the “_ S” (Underscore S) naming convention at the end. The reason we want to use both variables is that non-spend variables are a measure of exposure regardless of how much they cost. Spend and non-spend variables can have a complex relationship and usually fluctuate depending on numerous factors so it is important to use a variable more directly representing exposure to media.


Correct

```
set_mediaVarName <- c("tv_S"	,"ooh_S",	"print_S"	,"facebook_I"	,"search_clicks_P")

set_mediaSpendName <- c("tv_S"	,"ooh_S",	"print_S"	,"facebook_S"	,"search_S")
```

Incorrect

```
set_mediaVarName <- c("tv_S"	,"ooh_S",	"print_S"	,"facebook_I"	,"search_clicks_P")

set_mediaSpendName <- c("tv_S"	,"ooh_S",	"print_S")
```

#### set_mediaVarSign

You may apply sign control for media variables to be "default", "positive", or "negative". If you are expecting coefficients for baseline variables such as "tv", "print", "facebook" to be default (either positive or negative), positive or negative depending on its expected relationship with the dependent variable. We recommend using positive for all since media should have a positive effect on your dependent variable. Please remember the object declared must be same length as set_mediaVarName’s

```
set_mediaVarSign <- c("positive", "positive", "positive", "positive", "positive")

```

### Set factor variables

If any variable above should be the factor type, please include it in this section of the code, otherwise leave it empty as by default “c()” Variables that will be this type are variables that have qualitative data.

```
set_factorVarName <- c()
```

## Set global model parameters

In this section you will have to define parameters values and bounds for the
model to start working:

1. The **number of cores** in your computer to be used for **parallel
   computing**. It is  recommended to use less than 100% of the machine running the code’s cores. Please use detectCores() to find out the available cores in your machine.
```
   #### set model core features
   registerDoSEQ(); detectCores()
   set_cores <- 6
```

2. The **data training size** (set_modTrainSize) which will indicate the
   percentage of data to be used to train the model, therefore, the percentage
   left (1- training size) to validate the model.

   1. The function f.plotTrainSize helps you select the best split.     
      1. The function f.plotTrainSize helps you select the best split. Set the function to f.plotTrainSize(TRUE) to plot the Bhattacharyya coefficient, an indicator for the similarity of two distributions and goodness-of-fit, for the training size 50-90%. The coefficient can be used to determine the relative closeness of the samples being considered for test and validation.
      2. The higher the Bhattacharyya coefficient, the more similar the train and test data splits and thus the better the potential model fit in the end. The Bhattacharyya distance is widely used in research of feature extraction and selection. e.g. image processing.
```
      f.plotTrainSize(F) # insert TRUE to plot training size guidance.
      set_modTrainSize <- 0.74
```
3. The adstocking method, which can be
   [Geometric](https://en.wikipedia.org/wiki/Geometric_distribution) or
   [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) distributions.
```
      adstock <- "geometric"
```
4. The **number of iterations** (set_iter) your model will have to find optimum values for coefficients. If your objective is just to learn how the code works with the provided simulated data, you may reduce this number to get to results faster.
```
set_iter <- 500
```
5. After that, you will find set_hyperOptimAlgo <- "DiscreteOnePlusOne" which is the selected algorithm for Nevergrad, the gradient-free optimization library. There is no need to change the algorithm, however there are several to choose from.
```
set_hyperOptimAlgo <- "DiscreteOnePlusOne"
```
6. Finally, under ‘set_trial’ you will have to define the number of trials. 40 is recommended without calibration, 100 with calibration. If set_iter <- 500 and set_trial <- 40, this means that we will have 40 different initialization trials, each of them with 500 iterations. If your objective is just to learn how the code works with the provided simulated data, you may reduce this number to get to results faster.
```
set_trial <- 40
```

## Guidance to set hyperparameter bounds

The **hyperparameters bounds** which we recommend to leave as default but can be changed according to learnings from model iterations and analysts’ past experience.

   1. **The definition of each hyperparameter:**

      1. **Thetas**: Geometric function decay rate. For example, if the time unit for the model is weekly, it will represent the percentage of effect each week that is carried over to the next week.

      2. **Shapes**: Weibull parameter that controls the decay shape between exponential and s-shape. The larger, the more s-shape, the smaller, the more L-shape.

      3. **Scales**: Weibull parameter that controls the position of the decay inflection point. Recommended bounds are between 0 and 0.1. This is  because scale can inflate adstocking half-life siginificantly.

      4. **Alphas**: Hill function (Diminishing returns) parameter that controls the shape between exponential and s-shape. The larger the alpha, the more S-shape. The smaller, the more C-shape.

      5. **Gammas**: Hill function (Diminishing returns) parameter that controls the inflection point. The larger the gamma, the later the inflection point in the response curve.

   2. **Understanding how adstock affects media transformation:**
      1. In order to make more informed decisions to define hyperparameter values, it is very helpful to know which hyperparameter is doing what, during the media variables transformation. The plot function f.plotAdstockCurves helps you understand exactly that.
      2. Below we may find an example with different theta values for the geometric adstocking function. You may observe this is a one parameter (theta) function. Assuming the time unit is a week, we can see that when theta is 0.9, it means that 90% of the media effect each week is carried over to the next week. The adstocking halflife is reached after 8 weeks (halflife value in legend). In other words, it takes 8 weeks until the media effect decays to half when theta = 0.9. This should help you to have a more tangible feeling about values for theta and if they make sense for certain channels.

<img alt="adstockintro chart" src={useBaseUrl('/img/adstockintro.png')} />

Similar to the geometric function above, the Weibull plot visualises the two-parameter (scale & shape) from the Weibull function. The upper plot shows changes in scale while keeping shape constant. We can observe that the larger the scale, the later the inflection point. When scale=0.5 and shape = 2, it takes 18 weeks until the media effect decays to half (see legend). The lower plot shows changes in shape while keeping scale constant. When the shape is smaller, the curve rather takes an L-shape. When the shape is larger, the curve rather turns into an inverted S-shape.

<img alt="adstockintro2 chart" src={useBaseUrl('/img/adstockintro2.png')} />

The following media transformation bounds to set up are hill function’s (diminishing returns) curves scale and shape parameters. The theory of diminishing returns holds that each additional unit of advertising increases the response, but at a declining rate. This key marketing principle is reflected in marketing mix models as a variable transformation. You may observe below the differences in diminishing returns curves when changing alpha and gamma parameter ranges of values per channel. The higher the values for alpha, the more S-shape, the lower, the more C-shape.  Recommended ranges are  between 0.5 and 3 in order to provide reasonable curves that make sense for marketing variables. Whereas, the higher the values for gamma, the later the inflection point will appear. Late inflection points mean that the channel in question will have more room for investment and will reach a saturation point at higher investment levels.

<img alt="hillFunction1 chart" src={useBaseUrl('/img/hillFunction1.png')} />

## Accessing plotted results

Once all trials and iterations are finished, the model will proceed to plot different charts that will help you assess best models based on NRMSE and a media variables decomposition quality score for contribution of marketing channels. You may find all model plots like below example within the plot_folder you have set on the ‘run models’ step. You will see that there is a model unique ID associated with each model and chart saved in the folder. In the example below mod_ID = 1_22_3.

<img alt="ModelResults1 chart" src={useBaseUrl('/img/ModelResults1.png')} />

## Using the budget allocator

The budget allocator is also named optimizer. It provides the optimal media mix, which maximizes the return out of a certain spend level, while holding true to a set of constraints. Please note that the budget allocator will only output reasonable optimization when the MMM result makes sense, meaning all media channels have already found reasonable hyperparameters for adstocking and S-curving and the responses for each channel is meeting your expectations. Otherwise, the budget allocator output is not interpretable. Technically, the budget allocator consumes the response curve (Hill function) for each channel provided by the MMM result and conducts the solver for nonlinear optimization. The gradient-based algorithms (augmented Lagrangian / AUGLAG for global optimization and Method of Moving Asymptotes / MMA for local optimization) are chosen to solve the
nonlinear optimization problem with equal and unequal constraints. For details see [here](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/).

The first thing you will have to **define** is the **model unique ID** from the results in the previous section that you would like to use for the optimizer. Continuing with the example above, setting **modID = "1_22_3"** could be an example of a selected model from the list of best models in **'model_output_collect$allSolutions'** results object.

```
optim_result <- f.budgetAllocator(modID = "1_22_3"
                                ...
                                ...
)
```

Our current budget allocator has two scenarios:

- **Maximum historical response** (max_historical_response): Assuming two years of data for five media channels is used to build the model. Total spend was 1MM € with 40/30/15/10/5 split for both channels and total return was 2MM€. The budget allocator will output the optimum split for the historical spend level of 1MM€. For example, a maximum return of 2.5MM would be achieved with 35/25/20/12/8 split of the 1MM€ spent.

```
optim_result <- f.budgetAllocator(
                            modID = "1_22_3"
                          , scenario = "max_historical_response"
                            ...
                            ...
)
```

- **Maximum response of expected spend** (max_response_expected_spend): Compared to the above, this scenario outputs the optimum split of spend for a certain spend level, not the historical spend level. For example, if you have 100k€ for the next quarter, you would define expected_spend = 100000 and expected_spend_days = 90.

```
optim_result <- f.budgetAllocator(
                            modID = "1_22_3"
                          , scenario = "max_response_expected_spend"
                            ...
                            ...
)
```

For both scenarios, you must also define the constraints (lower and upper bounds) for each channel with the parameter channel_constr_low and channel_constr_up. Assuming for channel A you’ve spent 10k€ per week on average, then channel_constr_low = 0.7 and channel_constr_up = 1.2 will not allow the optimizer to go lower than 7k€ or higher than 12k€ for channel A when running the optimization. In general, please use realistic scenarios and avoid putting too extreme values. The budget allocator is still based on your historical performance.
For example, if you put 10 times as much as your historical spend, the budget allocator result may not make sense.

```
optim_result <- f.budgetAllocator(
                            modID = "1_22_3"
                          , scenario = "max_historical_response" # c(max_historical_response, max_response_expected_spend)
                          , channel_constr_low = c(0.7, 0.75, 0.60, 0.8, 0.65) # must be between 0.01-1 and has same length and order as set_mediaVarName
                          , channel_constr_up = c(1.2, 1.5, 1.5, 2, 1.5) # not recommended to 'exaggerate' upper bounds. 1.5 means channel budget can increase to 150% of current level
)
```

The result would look like the following. Again, we want to highlight that the result of the budget allocator will only be interpretable if the chosen model results make sense.

<img alt="budget allocator chart" src={useBaseUrl('/img/budgerAllocator1.png')} />
