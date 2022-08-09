# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn MMM Project from Meta Marketing Science
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
#' @author Igor Skokan (igorskokan@@fb.com)
#' @author Bernardo Lares (bernardolares@@fb.com)
#' @importFrom doRNG %dorng%
#' @importFrom doParallel registerDoParallel stopImplicitCluster
#' @importFrom dplyr any_of arrange as_tibble bind_rows contains desc distinct everything filter
#' group_by lag left_join mutate n pull rename row_number select slice summarise summarise_all ungroup
#' all_of bind_cols mutate_at tally
#' @importFrom foreach foreach %dopar% getDoParWorkers registerDoSEQ
#' @import ggplot2
#' @importFrom ggridges geom_density_ridges
#' @importFrom glmnet cv.glmnet glmnet
#' @importFrom jsonlite fromJSON toJSON write_json read_json
#' @importFrom lares check_opts clusterKmeans formatNum freqs glued ohse removenacols
#' theme_lares `%>%` scale_x_abbr scale_x_percent scale_y_percent scale_y_abbr try_require v2t
#' @importFrom lubridate is.Date day floor_date
#' @importFrom minpack.lm nlsLM
#' @importFrom nloptr nloptr
#' @importFrom parallel detectCores
#' @importFrom patchwork guide_area plot_layout plot_annotation wrap_plots
#' @importFrom prophet add_regressor fit.prophet prophet
#' @importFrom reticulate tuple use_condaenv import conda_create conda_install py_module_available
#' virtualenv_create py_install use_virtualenv
#' @importFrom rPref low psel
#' @importFrom stats AIC BIC coef end lm model.matrix na.omit nls.control median sd
#' predict pweibull dweibull quantile qunif reorder start setNames
#' @importFrom stringr str_count str_detect str_remove str_which str_extract str_replace
#' @importFrom tidyr pivot_longer pivot_wider
#' @importFrom utils askYesNo flush.console head setTxtProgressBar tail txtProgressBar write.csv
"_PACKAGE"

if (getRversion() >= "2.15.1") {
  globalVariables(c(".", "install_github"))
}
