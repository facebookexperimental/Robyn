---
id: calibration
title: Calibration using experimental results
---

import useBaseUrl from '@docusaurus/useBaseUrl';

### Calibration concept

By applying results from randomized controlled-experiments, you may improve the
accuracy of your marketing mix models dramatically. It is recommended to run
these on a recurrent basis to keep the model calibrated permanently. In general,
we want to compare the experiment result with the MMM estimation of a marketing
channel. Conceptually, this method is like a Bayesian method, in which we use
experiment results as a prior to shrink the coefficients of media variables. A
good example of these types of experiments is Facebook’s conversion lift tool
which can help guide the model towards a specific range of incremental values.

<img alt="Calibration chart" src={useBaseUrl('/img/calibration1.png')} />

The figure illustrates the calibration process for one MMM candidate model.
[**Facebook’s Nevergrad gradient-free optimization platform**](https://facebookresearch.github.io/nevergrad/) allows us to include the **MAPE(cal,fb)** as a third optimization score besides Normalized Root Mean Square Error (**NRMSE**) and **decomp.RSSD** ratio (Please refer to the automated hyperparameter selection and optimization for further details) providing a set of **Pareto optimal model solutions** that minimize and converge to a set of Pareto optimal model candidates. This calibration method can be applied to other media channels which run experiments, the more channels that are calibrated, the more accurate the MMM model.

### Calibration in the code

You will find a variable called 'activate_calibration' to be defined as T (True) or F (False). Please set it to T (True) if you would like to apply calibration.
Consequently, you will have to add ground-truth data such as conversion lift data from Facebook, geo tests or Multi-touch attribution. Moreover, you will need to define which channels you want to define certain incremental values for, as well as, the start date, end date and incremental absolute values (liftAbs) from the studies. In the example below there are two facebook studies with dates: from 2018-05-01 until 2018-06-10 and from 2018-07-01 until 2018-07-20. As well as, a TV geo study from 2017-11-27 until 2017-12-03. With a total lift in the absolute value for the response variable (Sales) of $400,000 for the first Facebook study, $300,000 for the TV study and of $200,000 for the last Facebook study.  

```
activate_calibration <- F # Switch to TRUE to calibrate model.
# set_lift <- data.table(channel = c("facebook_I",  "tv_S", "facebook_I"),
#                        liftStartDate = as.Date(c("2018-05-01", "2017-11-27", "2018-07-01")),
#                        liftEndDate = as.Date(c("2018-06-10", "2017-12-03", "2018-07-20")),
#                        liftAbs = c(400000, 300000, 200000))
```
