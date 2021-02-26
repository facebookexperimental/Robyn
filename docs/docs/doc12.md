---
id: doc12
title: Step-by-step guide
---

import useBaseUrl from '@docusaurus/useBaseUrl';

## Load data & scripts

The first step is to load data and scripts, you can start with our simulated
data file (de_simulated_data.csv) if you want to practice at the beginning.

```
script_path <- str_sub(rstudioapi::getActiveDocumentContext()$path, start = 1, end = max(unlist(str_locate_all(rstudioapi::getActiveDocumentContext()$path, "/"))))
dt_input <- fread(paste0(script_path,'de_simulated_data.csv'))
holidays <- fread(paste0(script_path,'generated_holidays.csv'))
```

Our script has been designed to pull files automatically from the folder where
the ‘fb_nextgen_mmm.exec.R’ script resides. If you would like to use a new data
input file instead of our simulated data you will need to save it in the same
folder and replace dt_input.csv file as in the example below: Before dt_input <-
fread(paste0(script_path,'de_simulated_data.csv')) After dt_input <-
fread(paste0(script_path,'your_input_data_file.csv'))

## Set model input variables

Once you have defined the input files to work with and loaded to the source all
the functions needed to run the code. You will need to define and set the input
variables.

### set_country

The first variable to declare is the country. We recommend using only one
country especially if you are planning to leverage prophet for trend and
seasonality which automatically pulls holidays for the country you have selected
and simplifies the process.

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

Setting the dependent variable is basically the outcome you are trying to
measure. We only accept one dependent variable under set_depVarName. This
variable can take the form of revenue (Sales or profit in monetary values) or
conversion (Number of transactions, units sold) which you will indicate when
defining the set_depVarType variable.

```
set_depVarName <- c("revenue") # there should be only one dependent variable
set_depVarType <- "revenue" # "revenue" or "conversion" are allowed
```

### Set Prophet variables

#### activate_prophet

First you will need to indicate the model if you would like to turn on or off
the Prophet feature in the code to be used for seasonality, trend and holidays.
T (True) means it is activated and F (False) deactivated.

```
activate_prophet <- T
```

#### set_prophet

The next step is to select which of the provided outcomes of Prophet you will
use in the model. It is recommended to at least keep trend and holidays. Please
have in mind that "trend","season", "weekday", "holiday" are provided and
case-sensitive.

```
set_prophet <- c("trend", "season", "holiday")
```

#### set_prophetVarSign

You may define the variable sign control for prophet variables to be "default",
"positive", or "negative". If you are expecting coefficients for prophet
variables such as "trend", "season", "holiday" to be default (either positive or
negative), positive or negative. We recommend using default. Please remember the
object declared must be same length as set_prophet’s

```
set_prophetVarSign <- c("default","default", "default")
```

#### Test and visualise prophet’s decomposition

You may drill-down into prophet results to understand better the outcomes
provided by the tool by executing the following command.

```
f.getProphet(dt_input)
```

### Set Baseline variables

The following step is to set the baseline variables which typically are things
like competitors, pricing, promotions, temperature, unemployment rate and any
other variable that is not media exposure but has a strong relationship with
sales outcomes. You will need to indicate the model if you would like to turn on
or off baseline variables in the code. T (True) means it is activated and F
(False) deactivated.

```
activate_baseline <- T
```

You may then define the different baseline variables you would like to consider.

```
set_baseVarName <- c("promotions", "price changes", "competitors sales")
```

You may apply sign control for baseline variables to be "default", "positive",
or "negative". If you are expecting coefficients for baseline variables such as
"promotions", "price changes", "competitors sales" to be default (either
positive or negative), positive or negative depending on its expected
relationship with the dependent variable. For example, rainy weather may have a
positive or negative impact in sales depending on the business. Please remember
the object declared must be same length as set_baseVarName’s

```
set_baseVarSign <- c("negative") # c("default", "positive", and "negative")
```

### set_mediaVarName and set_mediaSpendName

There is one key restriction to have in mind here, you must have spend variables
declared for every media channel you would like to measure. So they have to be
in the same order and same length as set_mediaVarName variables.

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

You may apply sign control for media variables to be "default", "positive", or
"negative". If you are expecting coefficients for baseline variables such as
"tv", "print", "facebook" to be default (either positive or negative), positive
or negative depending on its expected relationship with the dependent variable.
We recommend using positive for all. Please remember the object declared must be
same length as set_mediaVarName’s

```
set_mediaVarSign <- c("positive", "positive", "positive", "positive", "positive")
```

### Set factor variables

If any variable above should be factor please include it in this section of the
code, otherwise leave it empty as by default “c()”

```
set_factorVarName <- c()
```

## Set global model parameters

In this section you will have to define parameters values and bounds for the
model to start working:

1. The **number of cores** in your computer to be used for **parallel
   computing**
2. The **data training size** (set_modTrainSize) which will indicate the
   percentage of data to be used to train the model, therefore, the percentage
   left (1- training size) to validate the model.

   1. The function f.plotTrainSize helps you select the best split. Set the
      function to f.plotTrainSize(TRUE) to plot the Bhattacharyya coefficient,
      an indicator for the similarity of two distributions, for the training
      size 50-90%. The higher the Bhatta coefficient, the more similar the train
      and test data splits and thus the better the potential model fit in the
      end.

3. The adstocking method which can be
   [Geometric](https://en.wikipedia.org/wiki/Geometric_distribution) or
   [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) distributions.
4. The **number of iterations** (set_iter) your model will have to find optimum
   values for coefficients.
5. The **hyperparameters bounds** which we recommend to leave as default but can
   be changed according to learnings from model iterations and analysts’ past
   experience.

   1. **The definition of each hyperparameter:**

      1. **Thetas**: Geometric function decay rate
      2. **Shapes**: Weibull parameter that controls the decay shape between
         exponential and s-shape
      3. **Scales**: Weibull parameter that controls the position of inflection
         point
      4. **Alphas**: Hill function (Diminishing returns) parameter that controls
         the shape between exponential and s-shape
      5. **Gammas**: Hill function (Diminishing returns) parameter that controls
         the scale of transformation
      6. **Lambdas**: Regularization penalty parameter for ridge regression

   2. **Understanding how adstock affects media transformation:**
      1. In order to make more informed decisions to define hyperparameter
         values, it’s very helpful to know which hyperparameter is doing what
         during the media transformation. The plot function f.plotAdstockCurves
         helps you understand exactly that.
      2. Below is the geometric adstock that is a one parameter (theta)
         function. Assume the time unit is week, then we can see that when
         theta=0.9, meaning 90% of the media effect this week carries over to
         the next week, the halflife enters after 8 weeks (halflife value in
         legend). In other words, it takes 8 weeks until the media effect decays
         to half when theta=0.9. This should help you having a more tangible
         feeling about if the theta value for this certain channel makes normal
         sense.

<img alt="adstockintro chart" src={useBaseUrl('/img/adstockintro.png')} />

Similar to above, this plot visualises the two-parameter (scale & shape) Weibull
function. The upper plot shows changes in scale while keeping shape constant. We
can observe that the larger the scale, the later the inflexion point. When
scale=0.5 and shape = 2, it takes 18 weeks until the media effect decays to half
(see legend). The lower plot shows changes in shape while keeping scale
constant. When shape is smaller, the curve rather takes on the L-shape. When
shape is larger, the curve rather takes on S-shape.

<img alt="adstockintro2 chart" src={useBaseUrl('/img/adstockintro2.png')} />

In order to start setting model parameters we recommend to start with the
Geometric function.

```
## set model core features
adstock <- "geometric"
```

This function is simpler and therefore easier to understand compared to the
weibull function. You may start by focusing on Thetas’ bounds, which controls
for the decay rate any channel could take. If you are not sure about decay rates
for your channels, we suggest you start with a range: thetas = c(0, 0.5). This
means that in a week by week dataset, we would be expecting a maximum carryover
effect of 50% of previous week’s values to be extended and added to the
following week’s values for the dependent variable (Sales, impressions, etc.)

Once you run one of the iterations you may notice theta's values and if certain
optimums have been found by looking into variables including 'thetas' on their
names and identifying the number of times an optimum was found (epoch.optim).

<img alt="coderesults1 chart" src={useBaseUrl('/img/coderesults1.png')} />

In the example above, you will see that Facebook and TV thetas are still missing
an optimum given that optim.found column has still TRUE values on it, for this
we suggest to increase epoch number to Inf (Infinity) until an optimum is found.

```
model_output <- f.mmmRobyn(set_hyperBoundGlobal
                          ,set_iter = set_iter
                          ,set_cores = set_cores
                          ,epochN = Inf # set to Inf to auto-optimise until no optimum found
                          ,optim.sensitivity = 0 # must be from -1 to 1. Higher sensitivity means finding optimum easier
                          ,temp.csv.path = '/Users/leonelsentana/Documents/mmm.tempout.csv' # output optimisation result for each epoch. Use getwd() to find path
                          )
```

Below an example reaching optimum in 33 epochs:

<img alt="coderesults2 chart" src={useBaseUrl('/img/coderesults2.png')} />

You may note all parameters state FALSE which means there are not any more
optimums to find. This indicates that the values found are optimal, and that if
wanted, you could use some of these mode values to fix hyperparameter values in
the #### tune channel hyperparameters bounds section.

## Tune channel hyperparameters bounds

For that you will need to set the following command to TRUE:

```
activate_hyperBoundLocalTuning <- T
```

This command will enable you to define the bounds each media’s parameter should
have manually. Please uncomment the following section of the code and define the
values. Each bound can be either a range (e.g c(0.1,0.9)) or a fixed value:

```
 set_hyperBoundLocal <- list(
   facebook_I_alphas = c(0.8356271, 0.8371774)
   ,facebook_I_gammas = c(0.8991811, 0.9012183)
   ,facebook_I_thetas = c(0.8700282)
   ,ooh_S_alphas = c(0.8348503)
   ,ooh_S_gammas = c(0.3565417)
   ,ooh_S_thetas = c(0.3815053)
   ,print_S_alphas = c(0.6490568)
   ,print_S_gammas = c(0.1165910, 0.1171953)
   ,print_S_thetas = c(0.7617390)
   ,tv_S_alphas = c(2.4880460)
   ,tv_S_gammas = c(0.8867124)
   ,tv_S_thetas = c(0.1477954, 0.1478531)
   ,search_clicks_P_alphas = c(0.1360708)
   ,search_clicks_P_gammas = c(0.4177802)
   ,search_clicks_P_thetas = c(0.5784685)
 )
```

The hyperparameters tuning section is especially handy when we know there is an
expected behavior behind parameters e.g. adstocking for TV decay rate taking
only a value within a range. In this case, we would recommend you to guide the
model towards a solution that is both optimal and meets your needs based on
experience from previous analyses.

<img alt="coderesults2 chart" src={useBaseUrl('/img/coderesults2.png')} />

If we consider our previous example, it could be the case where you may want to
tune Theta decay rates to control for the adstocking decay pacing of TV to
**tv_S_thetas = c(0.25, 0.35)**. If we also fix other values where optimums have
been reached, we would reach something like the example below with fix values
and ranges based on previous iterations and past experience:

```
#Please uncomment set_hyperBoundLocal list completely and use scales and shapes only if you
#have selected ‘adstock <- "weibull"’. If you selected adstock <- "geometric" as by default, then simply #use alphas, gammas and thetas for each channel.

 set_hyperBoundLocal <- list(
   facebook_I_alphas = c(0.8356271, 0.8371774)
   ,facebook_I_gammas = c(0.8991811, 0.9012183)
   ,facebook_I_thetas = 0.8700282
   ,ooh_S_alphas = 0.8348503
   ,ooh_S_gammas = 0.3565417
   ,ooh_S_thetas = 0.3815053
   ,print_S_alphas = 0.6490568
   ,print_S_gammas = c(0.1165910, 0.1171953)
   ,print_S_thetas = 0.7617390
   ,search_clicks_P_alphas = 0.1360708
   ,search_clicks_P_gammas = 0.4177802
   ,search_clicks_P_thetas = 0.5784685
   ,tv_S_alphas = 2.4880460
   ,tv_S_gammas = 0.8867124
   ,tv_S_thetas = c(0.25, 0.35)
 )
```

If we apply the above bounds to our media related variables, you may see that
results now reach an optimum much faster, after only 2 epochs instead of 33. You
may also observe that the new bounds for **tv_S_thetas = c(0.25, 0.35)** have
been applied, therefore the new optimum found is 0.2706085 instead of the
0.1478256 from the previous example.

<img alt="coderesults3 chart" src={useBaseUrl('/img/coderesults3.png')} />

## Plotting and understanding results

The first thing you will have to do, is to set every plot you would like to be
drawn to TRUE (T):

```
#### Plot section

## insert TRUE into plot functions to plot
f.plotSpendModel(F)
f.plotHyperSamp(F, channelPlot = set_mediaVarName) # plot latin hypercube hyperparameter sampling balance. Max. 3 channels per plot
f.plotTrendSeason(F) # plot prophet trend, season and holiday decomposition
bestAdstock <- f.plotMediaTransform(T, channelPlot = set_mediaVarName) # plot best model media transformation with 3 plots: adstock decay rate, adstock effect & response curve. Max. 3 channels per plot
f.plotBestDecomp(F) # plot best model decomposition with 3 plots: sales decomp, actual vs fitted over time, & sales decomp area plot
f.plotMAPEConverge(F) # plot RS MAPE convergence, only for random search
f.plotBestModDiagnostic(F) # plot best model diagnostics: residual vs fitted, QQ plot and residual vs. actual
f.plotChannelROI(F)
f.plotHypConverge(F, channelPlot = set_mediaVarName) # plot hyperparameter vs MAPE convergence. Max. 3 channels per plot
boundOptim <- f.plotHyperBoundOptim(F, channelPlot = set_mediaVarName, model_output, kurt.tuner = optim.sensitivity)  # improved hyperparameter plot to better visualise trends in each hyperparameter
```

After that, you may execute each plot separately and analyze it.

### f.plotSpendModel: Plotting the spend-reach fitting with Michaelis-Menten model

The plot below illustrates how to better translate the relationship between
spend and reach so that you can apply it to performance ROI (Revenue/spend)
calculations. It is possible to choose reach variables (GRP, impressions, actual
reach, etc.) in the model. Understanding that relationship can help with the
translation of reach to spend and vice versa. LM (Linear Model) and NLS
(Non-linear least squares) comparison shows why it is better to use a non-linear
model, which is the technique applied in the code. The linear model is actually
as primitive as just using average to scale, because in our case the intercept
is omitted given that we assume no spend involves no reach. Whereas NLS are more
flexible and should adapt better to the underlying correlation patterns between
both variables.

<img alt="plotSpendModel chart" src={useBaseUrl('/img/plotSpendModel.png')} />

### f.plotHyperSamp: checking latin hypercube sampling distribution

The overall idea of these charts is to validate the **Latin hypercube sampling
(LHS)**, which is a statistical method for generating a near-random sample of
parameter values from the multidimensional distribution of hyperparameters. It
consists of two main charts:

- **Latin Hypercube Sampling distribution:** The key aspect is to concentrate on
  the randomness of the points on this chart. If all dots seem to cover evenly
  the space as in the example below, then sampling is ok. However, beware of
  high concentrations of dots or clear areas without any dots which may mean you
  will have to increase the sampling to have a better representation of the
  hyperspace.
- **Latin Hypercube Sampling distribution transformed:** This is a half violin
  chart that represents the distribution of each media parameter values. It
  helps you understand the actual range of values that each parameter is taking
  in order to represent uniformly the hyperparameter space. It will also depend
  on the set_hyperBoundGlobal variable and its bound definitions.

<img alt="plotHyperSamp chart" src={useBaseUrl('/img/plotHyperSamp.png')} />

### f.plotTrendSeason: Understanding trend and seasonality

This plot describes the trend, holidays and seasonality along the analyzed
period.

- The **trend** is the component of a time series that represents low frequency
  variations in time. It reflects the underlying tendency behind the dependent
  variable which can reflect growth, stability or shrinkage depending on the
  time unit analyzed.
- **Holidays** are basically days defined by governments where people
  commemorate events which can affect consumption behaviors. Some holidays may
  have more impact than others, therefore the different magnitudes on the chart.
- **Seasonality** is a characteristic of a time series in which the time series
  experiences regular and predictable changes that recur every calendar year.
  Any predictable fluctuation or pattern that recurs or repeats over a one-year
  period is said to be seasonal. Below we have the yearly seasonality as an
  example, which shows higher demand between October and December, and lower
  demand between April and July.

<img alt="plotTrendSeason chart" src={useBaseUrl('/img/plotTrendSeason.png')} />

### bestAdstock: Adstock carryover and diminishing returns effects

This plot represents the relationship between spend or ROI, and the response
variable (Sales, conversions, etc.). On the first line of charts you will find:
The response curve which can be S or C like, its scale of growth and diminishing
returns pace. The profit curve which illustrates the difference between the Y
and X axes (Response-profit). The higher this curve is on the Y axis, the closer
to a max profit point you will get. The second line of charts displays the
relationship between ROI and spend, the higher the curve on the Y axis, the
greater the response obtained. Maximum and Average ROI are drawn for an easier
read.

Note: This plot admits only 3 channels at a time. Please select the 3 channels
you would like to analyze as in the example below where first we look at channel
names, and second, we indicate which names within set_mediaVarName object to
plot c(1,2,5) :

```
set_mediaVarName
bestAdstock <- f.plotMediaTransform(T, channelPlot = set_mediaVarName[c(1,2,5)])
```

<img alt="bestAdstock chart" src={useBaseUrl('/img/bestAdstock.png')} />

### f.plotBestDecomp: Understanding the effect of baseline, media variables trend and seasonality along time

The aim of this chart is to display the breakdown of the response variable
through time and the relationship with each of its regressing components. As
with all marketing mix models, both media and non-media related explanatory
factors are displayed, including baseline effects such as price and promotions,
as well as, trend , seasonality and holidays. In order to be able to make a
proper description and forecast of performance, we must know to what extent each
component is present in the data.

<img alt="plotBestDecomp chart" src={useBaseUrl('/img/plotBestDecomp.png')} />

The chart below illustrates the contribution of each regressor variable to the
overall response (Sales in this case) You may find for example that Facebook
impressions had a 7.6% contribution to overall sales which represent in total
$28.8 million.

<img alt="bestDecomp2 chart" src={useBaseUrl('/img/bestDecomp2.png')} />

The next chart shows the difference between real and predicted values. The
closer the lines for y and y_pred, the better. You may also find RSQ and MAPE
metrics. RSQ or R-squared (R2) is a statistical measure that represents the
proportion of the variance for a dependent variable that is explained by an
independent variable or variables in a regression model. In general terms, the
closer to one, the better. Whereas, the mean absolute percentage error (MAPE) is
the mean or average of the absolute percentage errors of forecasts. Error is
defined as actual or observed value minus the forecasted value. As a general
rule, the closer to zero the better.

<img alt="bestDecomp3 chart" src={useBaseUrl('/img/bestDecomp3.png')} />

### f.plotMAPEConverge: Understanding MAPE evolution per Random Search iteration and minutes spent

The chart below displays the evolution of the mean absolute percentage error
(MAPE) with every minute spent on computing. This chart will help you understand
how many random search iterations to include in the model under **set_iter**
variable. If you increase set_iter from 100 to 10000, you will reach a point
where MAPE will not decrease in time. Therefore, finding the optimum may be
between 100 and 10000 may be your next step until you find the balance between
computing time and MAPE reduction.

<img alt="plotMAPEConverge chart" src={useBaseUrl('/img/plotMAPEConverge.png')}
/>

### f.plotBestModDiagnostic: plot best model diagnostics: residual vs fitted, QQ plot and residual vs. actual

Across these plots we analyze residuals, which are the difference between the
observed value of the dependent variable (y) and the predicted value (ŷ).

- **Fitted vs. Residual:** When conducting a residual analysis, a "residuals
  versus fitted plot" is the most frequently created plot. It is a scatter plot
  of residuals on the y axis and fitted values (estimated responses) on the x
  axis. The plot is used to detect non-linearity, unequal error variances, and
  outliers. The more random the pattern for the dots distribution in this plot
  and the more horizontal the smooth line, the better, as it would mean
  linearity, independent error variances and absence of outliers.

- **QQ Plot:** The quantile-quantile (q-q) plot is a graphical technique to
  determine if two data sets come from populations with a common distribution. A
  q-q plot is a plot of the quantiles of the first data set against the
  quantiles of the second data set in this case observed Y and Y_pred. A
  45-degree reference line is also plotted. The closer the dots to the 45
  degrees line, the better.

- **Observed vs. Residual:** Compared to the Fitted vs. Residual plot, this plot
  shows more correlation between observed/true response vs. residuals when the
  model R2 is lower. Normally, a certain degree of correlation here means either
  the data is just noisy or there’s still missing patterns within the error. In
  the context of MMM, we would recommend to assume the latter one and look for
  potential additional predictors to also capture these patterns. When
  conducting a residual analysis, a "residuals versus observed plot" is commonly
  used as a complement of residual vs. fitted plots. It is a scatter plot of
  residuals on the y axis and observed values (Actual responses) on the x axis.
  The plot is used to detect non-linearity, unequal error variances, and
  outliers. The more uniform the dots distribution in this plot and the more
  horizontal the smooth line, the better as it would mean independent error
  variances and absence of outliers. The contrast between fitted and observed
  vs. residual charts helps to understand if there are common relationships
  between both and residuals.

<img alt="f.plotBestModDiagnostic chart"
src={useBaseUrl('/img/f.plotBestModDiagnostic.png')} />

### f.plotChannelROI: Understanding performance with spend and return of investment charts

The plot below reflects a simple comparison of ROI and Spend per channel. The
greater the ROI (Return on investment) the better.

<img alt="plotChannelROI chart" src={useBaseUrl('/img/plotChannelROI.png')} />

### f.plotHypConverge: Correlation and convergence of hyperparameters and MAPE (Mean Absolute Percentage Error)

The plot below describes the relationship between mean absolute percentage error
(MAPE) and each hyperparameter range of values. It aims to assist you in finding
the right Hyperparameters bounds that will help on MAPE reduction, its minimum
values are indicated with a vertical line.

<img alt="plotHypConverge chart" src={useBaseUrl('/img/f.plotHypConverge.png')}
/>

### f.plotHyperBoundOptim: Improved plot for convergence of hyperparameters and MAPE (Mean Absolute Percentage Error)

Similar to the previous plot, this plot illustrates the relationship between
mean absolute percentage error (MAPE) and each hyperparameter range of values
via kurtosis distributions. It aims to assist you in finding the right
Hyperparameters bounds that will help on MAPE reduction, its minimum values are
indicated with a vertical line, in this case the mode. The mode can be taken as
the most frequent value, therefore indicating, together with low and up vertical
lines, the possible bounds to be adjusted on **set_hyperBoundLocal** variable

<img alt="f.plotHyperBoundOptim chart"
src={useBaseUrl('/img/f.plotHyperBoundOptim.png')} />

## Using the optimiser

Optimiser is also named scenario planner. It gives you the optimum media mix,
meaning getting the most return out of a certain spend level, while holding true
to a set of constraints. Please note that the optimiser will only output
reasonable optimisation when the MMM result makes sense, meaning all media
channels have already found reasonable hyperparameters for adstocking and
S-curving and the responses for each channel is meeting your expectations.
Otherwise, the optimiser output is not interpretable. Technically, the optimiser
consumes the response curve (Hill function) for each channel provided by the MMM
result and conducts the solver for nonlinear optimisation. The gradient-based
algorithms (augmented Lagrangian / AUGLAG for global optimisation and Method of
Moving Asymptotes / MMA for local optimization) are chosen to solve the
nonlinear optimisation problem with equal and unequal constraints. For details
see [here](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/).

Our current optimiser has two scenarios:

- Maximum historical response (max_hostorical_response): Assuming two years of
  data for five media channels is used to build the model. Total spend was 1MM €
  with 40/30/15/10/5 split for both channels and total return was 2MM€. The
  optimiser will output the optimum split for the historical spend level of
  1MM€. For example, a maximum return of 2.5MM would be achieved with
  35/25/20/12/8 split of the 1MM€ spent.

<img alt="optimiser1 chart" src={useBaseUrl('/img/optimiser1.png')} />

- Maximum response of expected spend (max_response_expected_soend): Compared to
  the above, this scenario outputs the optimum split of spend for a certain
  spend level, not the historical spend level. For example, if you have 100k€
  for the next quarter, you would define expected_spend = 100000 and
  expected_spend_days = 90.

<img alt="optimiser2 chart" src={useBaseUrl('/img/optimiser2.png')} />

For both scenarios, you must also define the constraints (lower and upper
bounds) for each channel with the parameter channel_constr_low and
channel_constr_up. Assuming for channel A you’ve spent 10k€ per week on average,
then channel_constr_low = 0.7 and channel_constr_up = 1.2 will not allow the
optimiser to go lower than 7k€ or higher than 12k€ for channel A when running
the optimisation. In general, please use realistic scenarios and avoid putting
too extreme values. The optimiser is still based on your historical performance.
For example, if you put 10 times as much as your historical spend, the optimiser
result might not be making sense.

The result would look like the following. Again, we want to address that the
result of optimiser will only be interpretable if the MMM result makes sense.

<img alt="optimiser3 chart" src={useBaseUrl('/img/optimiser3.png')} />
