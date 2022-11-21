#### pareto_aggregated.csv: Aggregated result for all independent variables

# $ solID : Character. Model ID of pareto output.
# $ rn : Character. Independent variable names
# $ coef : Numeric. estimated beta coefficient
# $ xDecompAgg : Numeric. Aggregated decomposed effect. Pseudo-calc: sum(beta1 * channel1_adstocked_saturated)
# $ xDecompPerc : Numeric. Share of decomposed effect. Pseudo-calc: xDecompAgg / sum(xDecompAgg)
# $ xDecompMeanNon0 : Numeric. Mean decomposed effect of non-zero spend periods. Pseudo-calc: mean(beta1 * channel1_adstocked_saturated[spend > 0])
# $ xDecompMeanNon0Perc : Numeric. Share of xDecompMeanNon0. Pseudo-calc: xDecompMeanNon0 / sum(xDecompMeanNon0)
# $ xDecompAggRF : Numeric. xDecompAgg for the appended refreshing period only.
# $ xDecompPercRF : Numeric. Share of xDecompAgg.
# $ xDecompMeanNon0PercRF: Numeric. xDecompMeanNon0 for the appended refreshing period only.
# $ pos : Logical. coef sign check. TRUE when coef >= 0
# $ spend_share_refresh : Numeric. Share of total spend for the appended refreshing period only.
# $ effect_share_refresh : Numeric. Share of total effect for the appended refreshing period only.
# $ mape : Numeric. MAPE.LIFT for calibration is the third objective function when using calibration.
# $ nrmse : Numeric. NRMSE normalised root-mean of squared error. Main objective function.
# $ decomp.rssd : Numeric. DECOMP.RSSE decomposition root-sum of squared error. Main objective function.
# $ rsq_train : Numeric. Adjusted R squared of training data.
# $ lambda : Numeric. The L2 regularization parameter.
# $ iterPar : Integer. Parallel iteration in inner loop of robyn_mmm that equals number of cores.
# $ iterNG : Integer. Pseudo-calc: iterNG = round(total_iteration / iterPar)
# $ df.int : Integer. Degree of freedom for intercept that takes on 0 or 1.
# $ trial : Integer. Trial of current model.
# $ iterations : Integer. Iteration position of current model and current trial.
# $ robynPareto : Integer. Position of pareto front of current model.
# $ total_spend : Numeric. Total spend of each paid_media_vars.
# $ mean_spend : Numeric. Mean spend of non-zero periods of each paid_media_vars.
# $ spend_share : Numeric. Share of total_spend.
# $ effect_share : Numeric. Share of total effect among paid media. Pseudo-calc: xDecompPerc / sum(xDecompPerc) for paid media only
# $ roi_mean : Numeric. Pseudo-calc: roi_mean = mean_response / mean_spend.
# $ roi_total : Numeric. Pseudo-calc: roi_total = xDecompAgg / total_spend.
# $ cpa_total : Numeric. Pseudo-calc: cpa_total = total_spend / xDecompAgg
# $ mean_response : Numeric. Response of mean_spend. Pseudo-calc: mean_response1 = beta1 * saturated(mean_spend1) Note the difference to xDecompMeanNon0.
# $ next_unit_response : Numeric. Response of next unit spend from the level of mean_spend. Pseudo-calc: next_unit_response1 = beta1 * (saturated(mean_spend1 + 1) - saturated(mean_spend1))
# $ cluster : Integer. Cluster index of the model
# $ top_sol : Logical. TRUE indicates the model is selected as cluster winner
# $ boot_mean : Numeric. Mean of bootstrapped in-cluster CI
# $ boot_se: Numeric. Standard error of bootstrapped in-cluster CI
# $ ci_low: Numeric. Lower bound of bootstrapped in-cluster CI adapted for the sample
# $ ci_up: Numeric. Lower bound of bootstrapped in-cluster CI adapted for the sample
# $ carryover_pct: Numeric. Share of carryover response from total. 0.7 means 70% of total response comes from carryover.

#### pareto_hyperparameters.csv: Value of all hyperparameters of all pareto models. Number of column varies depending on input data

# $ solID : Character. Model ID of pareto output.
# $ channel_1_alphas : Numeric. value of alpha of channel1
# $ channel_1_gammas : Numeric. value of gamma of channel1
# $ ...
# $ mape	: see schema of pareto_aggregated.csv
# $ nrmse	: see schema of pareto_aggregated.csv
# $ decomp.rssd	: see schema of pareto_aggregated.csv
# $ rsq_train	: see schema of pareto_aggregated.csv
# $ pos	: see schema of pareto_aggregated.csv
# $ lambda	: see schema of pareto_aggregated.csv
# $ Elapsed	: Numeric. Elapsed time per iteration
# $ ElapsedAccum : Numeric. Accumulated elapsed time per iteration
# $ iterPar	: see schema of pareto_aggregated.csv
# $ iterNG	: see schema of pareto_aggregated.csv
# $ df.int	: see schema of pareto_aggregated.csv
# $ trial	: see schema of pareto_aggregated.csv
# $ iterations	: see schema of pareto_aggregated.csv
# $ coef0	: Logical. TRUE when current model contains channel with coef == 0
# $ mape.qt10: Logical. TRUE when the calibration error MAPE.LIFT of current model is among lowest 10% of all iterations.
# $ robynPareto : see schema of pareto_aggregated.csv

#### pareto_alldecomp_matrix.csv: Time series vectors of predicted response of all variables. Number of column varies depending on input data.

# $ ds : Date. Date stamp in format '2022-01-01'
# $ dep_var : Numeric. Raw dependent variable.
# $ trend : Numeric. Vector of predicted response of trend.
# $ season : Numeric. Vector of predicted response of season.
# $ weekday : Numeric. Vector of redicted response of weekday.
# $ channel_1 : Numeric. Vector of redicted response of channel1.
# $ channel_2: Numeric. Vector of redicted response of channel1.
# $ ...
# $ intercept : Numeric. Vector of redicted response of intercept
# $ depVarHat : Numeric. Vector of total predicted response. Pseudo-calc: depVarHat = sum(trend, season, channel1 ...)
# $ solID : Character. Model ID of pareto output.

#### pareto_media_transform_matrix.csv: Time series vectors of all raw and transformed paid & organic media variables. Number of column varies depending on input data.

# $ ds : Date. Date stamp in format '2022-01-01'
# $ channel_1 : Numeric. Vector of redicted response of channel1.
# $ channel_2: Numeric. Vector of redicted response of channel1.
# $ ...
# $ type : Character. Type of transformation (raw, adstocking, saturation, final response).
# $ solID : Character. Model ID of pareto output.


#### All _reallocated.csv as well as AllocatorCollect$dt_optimOut for each model.

# $ channels : Character. Equals to paid_media_vars. Those with coefficient = 0 are excluded in allocator because it doesn't make sense to optimise 0 response channel.
# $ histSpend : Numeric. Total spend of each channel within modelling window.
# $ histSpendTotal : Numeric. Sum of histSpend, including excluded channels. This also represents total historical marketing budget.
# $ initSpendUnitTotal : Numeric. Sum of initSpendUnit, including excluded channels.
# $ initSpendUnit : Numeric. Mean spend of each channel for the respective time grain. Note that the mean is calculated with non-zero spend period. For example, when a new channel has spend c(0, 0, 0, 5, 5), mean spend is 10/2 = 5, because we want mean spend to reflect active spending decision of marketing.
# $ initSpendShare : Numeric. Equals initSpendUnit / initSpendUnitTotal. Sum of initSpendShare is <1 when 0-coef channels are excluded.
# $ initResponseUnit : Numeric. Response of initSpendUnit of each channel for the respective time grain.
# $ initResponseUnitTotal : SNumeric. um of initResponseUnit. Excluded channels don't impact it here because they have 0 response anyway.
# $ initRoiUnit : Numeric. Equals initResponseUnit / initSpendUnit
# $ expSpendTotal : Numeric. Total marketing budget for optimal allocation. When scenario = "max_historical_response", it equals histSpendTotal. When scenario = "max_response_expected_spend", it equals to user's input in expected_spend argument in robyn_allocator() function
# $ expSpendUnitTotal : Numeric. Total marketing budget for optimal allocation for the respective time grain. When scenario = "max_historical_response", it equals to initSpendUnitTotal. When scenario = "max_response_expected_spend", it equals to expected_spend / expected_days
# $ expSpendUnitDelta : Numeric. Equals expSpendUnitTotal / initSpendUnitTotal -1
# $ optmSpendUnit : Numeric. Optimised spend of each channel for the respective time grain.
# $ optmSpendUnitDelta : Numeric. Equals optmSpendUnit / initSpendUnit -1. This also reflects the channel constraints in channel_constr_low and channel_constr_up.
# $ optmSpendUnitTotal : Numeric. Equals sum(optmSpendUnit). When scenario = "max_historical_response", it equals to initSpendUnitTotal and expSpendUnitTotal. However when channel_constr_up is too restrictive, it might be lower that the latter two metrics. When scenario = "max_response_expected_spend", it equals to expected_spend / expected_days. However when channel_constr_up is too restrictive, it might be lower that the latter one.
# $ optmSpendUnitTotalDelta : Numeric. Equals optmSpendUnitTotal / initSpendUnitTotal -1.
# $ optmSpendShareUnit : Numeric. Equals optmSpendUnit / sum(optmSpendUnit)
# $ optmResponseUnit : Numeric. Response level of optmSpendUnit for each channel
# $ optmResponseUnitTotal : Numeric. Equals sum(optmResponseUnit)
# $ optmRoiUnit : Numeric. Equals optmResponseUnit / optmSpendUnit
# $ optmResponseUnitLift : Numeric. Equals optmResponseUnit / initResponseUnit - 1
# $ optmResponseUnitTotalLift: Numeric. Equals optmResponseUnitTotal / initResponseUnitTotal - 1
