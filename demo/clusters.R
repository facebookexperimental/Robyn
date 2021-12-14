# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# LOAD LIBRARIES
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(lares))

# LOAD DATA
load("~/Desktop/Robyn-outputs/Robyn.RData")
df <- read.csv("~/Desktop/Robyn-outputs/2021-08-25 12.25 init/pareto_aggregated.csv")
hyps <- read.csv("~/Desktop/Robyn-outputs/2021-08-25 12.25 init/pareto_hyperparameters.csv")

# TRANSFORM DATA
ids <- "solID"
dfp <- pivot_wider(df, id_cols = ids, names_from = rn, values_from = roi_total)
dfp <- removenacols(dfp, all = FALSE)
dfp <- select(dfp, any_of(c(ids, Robyn$listInit$InputCollect$all_media)))
# Gather the errors data
errors <- distinct(df, solID, nrmse, decomp.rssd, mape)
dfp <- left_join(dfp, errors, "solID")
# Test adding hyperparameters data to compare clusters
hyp <- select(hyps, .data$solID, contains(Robyn$listInit$InputCollect$all_media))
dfp <- left_join(dfp, hyp, "solID")
# Check the data format and information available
head(dfp)
glimpse(dfp)

# 1. SELECT: NUMBER OF CLUSTERS
# ignore <- unique(c(ids, names(errors), "pareto")) # keep hyperparameters?
ignore <- unique(c(ids, names(errors), "pareto", names(hyp)[-1]))
cls <- clusterKmeans(dfp, ignore = ignore)
cls$nclusters_plot

################################################################
n_clusters <- 6 # Set as default in Robyn's function: N = 6
################################################################

# CHECK EACH CLUSTERS CHARACTERISTICS
cls <- clusterKmeans(dfp, n_clusters, dim_red = "PCA", ignore = ignore)
cls$nclusters_plot
cls$correlations
cls$means
# Dim reduction plots
cls$PCA$plot

# # Are hyperparameters similar within clusters? With our without hypers data used.
# cls$df %>% select(.data$solID, .data$cluster) %>%
#   left_join(hyp, "solID") %>%
#   mutate(cluster = cls$df$cluster) %>%
#   group_by(cluster) %>% mutate(n = n(), cluster = sprintf("%s (%s)", .data$cluster, .data$n)) %>%
#   tidyr::gather("hyperparameter", "value", any_of(names(hyp)[-1])) %>%
#   ggplot(aes(x = hyperparameter, y = value, fill = cluster)) +
#   geom_boxplot() + #facet_grid(cluster~.) +
#   coord_flip() + theme_lares(pal = 1) +
#   labs(title = "Are hyperparameters similar within clusters?",
#        subtitle = "Without using hyperparameters data")

# # Mean values don't give overall trend on clusters. Use correlation instead!
# cls$means %>%
#   tidyr::gather("media", "mean_roi", any_of(Robyn$listInit$InputCollect$all_media)) %>%
#   ggplot(aes(x = .data$media, y = .data$mean_roi)) +
#   facet_grid(.data$cluster ~ .) +
#   geom_col() + coord_flip()

################################################################
# SELECT CRITERIA TO SELECT N TOP MODELS BY CLUSTER
limit <- 2 # N top solutions per cluster
balance <- c(1, 1, 1) # Non-normalized weights for errors - size 3
################################################################

# Auxiliary functions
min_max_norm <- function(x) (x - min(x)) / (max(x) - min(x))
crit_df <- function(df, balance = rep(1, 3)) {
  stopifnot(length(balance) == 3)
  balance <- balance/sum(balance)
  crit_df <- df %>%
    # Force normalized values so they can be comparable
    mutate(nrmse = min_max_norm(.data$nrmse),
           decomp.rssd = min_max_norm(.data$decomp.rssd),
           mape = min_max_norm(.data$mape)) %>%
    # Balance to give more or less importance to each error
    mutate(nrmse = balance[1]*.data$nrmse,
           decomp.rssd = balance[2]*.data$decomp.rssd,
           mape = balance[3]*.data$mape) %>%
    replace(., is.na(.), 0) %>%
    group_by(.data$cluster)
  return(crit_df)
}
crit_proc <- function(df, limit) {
  arrange(df, .data$cluster, desc(.data$error)) %>%
    slice(1:limit) %>%
    mutate(rank = row_number()) %>%
    select(.data$cluster, .data$rank, everything())
}
plot_topsols_errors <- function(df, top_sols, criteria = "Criteria", limit = 1, balance = rep(1, 3)) {
  balance <- balance/sum(balance)
  left_join(df, select(top_sols, 1:3), "solID") %>%
    mutate(alpha = ifelse(is.na(.data$cluster), 0.5, 1),
           label = ifelse(!is.na(.data$cluster), sprintf(
             "[%s.%s]", .data$cluster, .data$rank), NA)) %>%
    ggplot(aes(x = .data$nrmse, y = .data$decomp.rssd)) +
    geom_point(aes(colour = .data$cluster, alpha = .data$alpha)) +
    geom_text(aes(label = .data$label), na.rm = TRUE, hjust = -0.3) +
    guides(alpha = "none", colour = "none") +
    labs(title = paste("Selecting Top", limit, "Performing Models by Cluster"),
         subtitle = paste("Criteria:", criteria),
         x = "NRMSE", y = "DECOMP.RSSD",
         caption = sprintf("Weights: NRMSE %s%%, DECOMP.RSSD %s%%, MAPE %s%%",
                           round(100*balance[1]), round(100*balance[2]), round(100*balance[3]))) +
    theme_lares()
}
plot_topsols_rois <- function(top_sols, limit = 1) {
  top_sols %>%
    select(-contains(names(hyp)[-1])) %>%
    mutate(label = sprintf("Cl. %s\n%s", .data$cluster, .data$solID)) %>%
    tidyr::gather("media", "roi", contains(Robyn$listInit$InputCollect$all_media)) %>%
    ggplot(aes(x = .data$media, y = .data$roi)) +
    facet_grid(.data$label ~ .) +
    geom_col() + coord_flip() +
    labs(title = paste("ROIs on Top", limit, "Performing Models by Cluster"),
         x = NULL, y = "ROI per Media") +
    theme_lares()
}

criteria <- "By minimum weighted distance to zero"
top_sols <- crit_df(cls$df, balance) %>%
  mutate(error = (nrmse^2 + decomp.rssd^2 + mape^2)^-(1/2)) %>%
  crit_proc(limit); top_sols
plot_topsols_errors(dfp, top_sols, criteria, limit, balance)
plot_topsols_rois(top_sols, limit)
