# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn MMM Open Source 3.0 - a Beta Project from Facebook Marketing Science
#'
#' Robyn is an automated Marketing Mix Modeling (MMM) code. It aims to reduce human
#' bias by means of ridge regression and evolutionary algorithms, enables actionable
#' decision making providing a budget allocator and diminishing returns curves and
#' allows ground-truth calibration to account for causation.
#'
#' @md
#' @name Robyn
#' @docType package
#' @author Gufeng Zhou (gufeng@@fb.com)
#' @author Leonel Sentana (leonelsentana@@fb.com)
#' @author Antonio Prada (aprada@@fb.com)
#' @author Igor Skokan (igorskokan@@fb.com)
#' @import data.table
#' @importFrom doFuture registerDoFuture
#' @importFrom doRNG %dorng%
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach foreach %dopar% getDoParWorkers registerDoSEQ
#' @importFrom future multicore plan sequential availableCores
#' @import ggplot2
#' @importFrom glmnet cv.glmnet glmnet
#' @importFrom lubridate is.Date day floor_date
#' @importFrom minpack.lm nlsLM
#' @importFrom nloptr nloptr
#' @importFrom parallel detectCores
#' @importFrom patchwork guide_area plot_layout plot_annotation wrap_plots
#' @importFrom prophet add_regressor fit.prophet prophet
#' @importFrom reticulate tuple use_condaenv import conda_create conda_install py_module_available
#' @importFrom rPref low psel
#' @importFrom stats AIC BIC coef end lm model.matrix na.omit nls.control
#' predict pweibull quantile qunif start
#' @importFrom stringr str_detect str_remove str_which str_extract str_replace
#' @importFrom utils askYesNo head setTxtProgressBar txtProgressBar
"_PACKAGE"

# data.table column names used
dt_vars <- c(
  "Elapsed", "ElapsedAccum", "Km", "Vmax", "actual", "avg_decay_rate", "bestModRF", "channel",
  "channels", "country", "cpa_total", "decay_accumulated", "decomp.rssd", "decompAbsScaled",
  "decomp_perc", "decomp_perc_prev", "depVarHat", "dep_var", "ds", "dsMonthStart", "dsWeekStart",
  "duration", "effect_share", "effect_share_refresh", "error_dis", "exposure", "halflife",
  "holiday", "i.effect_share_refresh", "i.robynPareto", "i.solID", "i.spend_share_refresh",
  "id", "initResponseUnit", "initResponseUnitTotal", "initSpendUnit", "iterNG", "iterPar",
  "iterations", "label", "liftAbs", "liftEndDate", "liftMedia", "liftStart", "liftStartDate",
  "mape", "mape.qt10", "mape_lift", "mean_response", "mean_spend", "mean_spend_scaled",
  "models", "next_unit_response", "nrmse", "optmResponseUnit", "optmResponseUnitTotal",
  "optmResponseUnitTotalLift", "optmSpendUnit", "optmSpendUnitTotalDelta", "param",
  "perc", "percentage", "pos", "predicted", "refreshStatus", "response", "rn", "robynPareto",
  "roi", "roi_mean", "roi_total", "rsq_lm", "rsq_nls", "rsq_train", "s0", "scale_shape_halflife",
  "season", "sequential", "shape", "solID", "spend", "spend_share", "spend_share_refresh",
  "theta", "theta_halflife", "total_spend", "trend", "trial", "type", "value", "variable",
  "weekday", "x", "xDecompAgg", "xDecompMeanNon0", "xDecompMeanNon0Perc",
  "xDecompMeanNon0PercRF", "xDecompMeanNon0RF", "xDecompPerc", "xDecompPercRF", "y", "yhat"
)

if (getRversion() >= "2.15.1") {
  globalVariables(c(".", dt_vars))
}
