---
id: doc8
title: Calibration using experimental results
---
import useBaseUrl from '@docusaurus/useBaseUrl';

### Calibration concept
By applying results from randomized controlled-experiments, you may improve the accuracy of your marketing mix models dramatically. It is recommended to run these on a recurrent basis to keep the model calibrated permanently. In general, we want to compare the experiment result with the MMM estimation of a marketing channel. Conceptually, this method is like a Bayesian method, in which we use experiment results as a prior to shrink the coefficients of media variables. A good example of these types of experiments is Facebook’s conversion lift tool which can help guide the model towards a specific range of incremental values.

<img alt="Calibration chart" src={useBaseUrl('/static/img/calibration1.png')} />


Figure illustrates the calibration process above for one MMM candidate model. 
Table below illustrates the model selection output including FB lift calibration element. Modelers can select the top models with relatively small MAPE metrics as the candidates for the final model. In this example, we suggest picking model two, as it has the minimum <em>MAPE(cal,fb)</em> and its <em>MAPE(holdout)</em> is only 0.4% more than the minimum one.

#### Example Table 
Sample output of model selection of a MMM with only two media channels, TV and Social
<img alt="Calibration table" src={useBaseUrl('/static/img/calibration2.png')} />

Note that <em>MAPE(cal,fb)</em> will likely vary more widely than <em>MAPE(holdout)</em> . Given this, calibration can improve performance without substantially sacrificing backtesting performance. 
This calibration method can be applied to other media channels which run experiments, the more channels that are calibrated, the more accurate the MMM model. <em>You may find the calibration function in the ‘func.R’ script.</em> 

### Calibration in the code

So, how do we apply this in our code?

1. First, we check if media channels to be calibrated actually have a media variable created. 
2. After that, we collect all different media to be calibrated. Consequently, we loop over each lift channel (Where for each of them we iterate over all different studies if more than one, determining the date range of each study) 
3. In addition, we convert data from weeks to days (Please note the *7 in the formula for mmmDays, this is assuming you will use weekly data as a basis for your model). 
4. Finally, and once both lift study and MMM dates are both in days, we scale the total decomposed model predicted sales into the exact number of days the lift study had to be comparable with previously uploaded liftAbs number under the set_lift variable (remember liftAbs values in set_lift variable have to be absolute and measuring the same metric as the model does ie. total incremental sales vs. model predicted sales)

```
#### Define lift calibration function
f.calibrateLift <- function(decompCollect, set_lift) {
  
  check_set_lift <- any(sapply(set_lift$channel, function(x) any(str_detect(x, set_mediaVarName)))==F) #check if any lift channel doesn't have media var
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
      liftPeriodVec <- getDecompVec[DS >= liftStart & DS <= liftEnd, c("DS", getLiftMedia[m]), with = F]
      
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
```

The last step is to calculate the MAPE. This will be the key metric to define the model that is closest to actual incremental sales during periods for the lift study. It will therefore allow us to make a decision as per the example on the [**table**](#example-table).
```
  ## get mape_lift
  liftCollect <- rbindlist(liftCollect)[, mape_lift := abs((decompAbsScaled - liftAbs) / liftAbs) * 100]
  return(liftCollect) 
}
```