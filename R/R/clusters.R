# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Reduce number of models based on ROI clusters and minimum combined errors
#'
#' The \code{robyn_clusters()} function uses output from \code{robyn_run()},
#' to reduce the amount of models and help the user pick up the best (lowest
#' combined error) of different kinds (clusters) of models.
#'
#' @inheritParams lares::clusterKmeans
#' @inheritParams hyper_names
#' @param input \code{robyn_run()}'s output or \code{pareto_aggregated.csv} results.
#' @param limit Integer. Top N results per cluster.
#' @param weights Vector, size 3. How much should each error weight?
#' Order: nrmse, decomp.rssd, mape. The highest the value, the closer it will be scaled
#' to origin (ei/x). Each value will be normalized so they all sum 1.
#' @param ... Additional parameters passed to \code{lares::clusterKmeans()}.
#' @author Bernardo Lares (bernardolares@@fb.com)
#' @examples
#' \dontrun{
#' cls <- robyn_clusters(input = OutputCollect,
#'                       all_media = InputCollect$all_media,
#'                       k = 3, limit = 2,
#'                       weights = c(1, 1, 1.5))
#' }
#' @export
robyn_clusters <- function(input, all_media = NULL,
                           k = 6, limit = 1,
                           weights = rep(1, 3),
                           dim_red = "PCA",
                           ...) {
  df <- attr(input, "ROIs")
  if (is.null(df)) {
    if (all(c("solID", "mape", "nrmse", "decomp.rssd") %in% names(input)) & is.data.frame(input)) {
      df <- .prepare_roi(input, all_media)
    } else {
      stop(paste(
        "You must run robyn_run(..., clusters = TRUE) or",
        "pass a valid data.frame (sames as pareto_aggregated.csv output)",
        "in order to use robyn_clusters()"
      ))
    }
  } else {
   if (is.null(all_media)) {
     aux <- colnames(input$mediaVecCollect)
     all_media <- aux[-c(1, which(aux == "type"):length(aux))]
   }
  }

  # Build clusters
  ignore <- c("solID", "mape", "decomp.rssd", "nrmse", "pareto")
  cls <- clusterKmeans(df, k, ignore = ignore, dim_red = dim_red, quiet = TRUE, ...)

  # Select top models by minimum (weighted) distance to zero
  top_sols <- .clusters_df(cls$df, weights) %>%
    mutate(error = (.data$nrmse^2 + .data$decomp.rssd^2 + .data$mape^2)^-(1 / 2)) %>%
    .crit_proc(limit)

  output <- list(
    # Within Groups Sum of Squares Plot
    clusters_wgss = cls$nclusters_plot,
    # Grouped correlations per cluster
    clusters_corrs = cls$correlations,
    # Mean ROI per cluster
    clusters_means = cls$means,
    # Dim reduction clusters
    clusters_PCA = cls[["PCA"]],
    clusters_tSNE = cls[["tSNE"]],
    # Top Clusters
    clusters_models = top_sols,
    clusters_models_errors = .plot_topsols_errors(df, top_sols, limit, weights),
    clusters_models_rois = .plot_topsols_rois(top_sols, all_media, limit),
    # Parameters
    n_clusters = k,
    errors_weights = weights
  )
  return(output)
}


# ROIs data.frame for clustering (from xDecompAgg or pareto_aggregated.csv)
.prepare_roi <- function(x, all_media) {
  check_opts(all_media, unique(x$rn))
  rois <- pivot_wider(x, id_cols = "solID", names_from = "rn", values_from = "roi_total")
  rois <- removenacols(rois, all = FALSE)
  rois <- select(rois, any_of(c("solID", all_media)))
  errors <- distinct(x, .data$solID, .data$nrmse, .data$decomp.rssd, .data$mape)
  rois <- left_join(rois, errors, "solID")
  return(rois)
}

.min_max_norm <- function(x) (x - min(x)) / (max(x) - min(x))

.clusters_df <- function(df, balance = rep(1, 3)) {
  stopifnot(length(balance) == 3)
  balance <- balance / sum(balance)
  crit_df <- df %>%
    # Force normalized values so they can be comparable
    mutate(
      nrmse = .min_max_norm(.data$nrmse),
      decomp.rssd = .min_max_norm(.data$decomp.rssd),
      mape = .min_max_norm(.data$mape)
    ) %>%
    # Balance to give more or less importance to each error
    mutate(
      nrmse = balance[1] / .data$nrmse,
      decomp.rssd = balance[2] / .data$decomp.rssd,
      mape = balance[3] / .data$mape
    ) %>%
    replace(., is.na(.), 0) %>%
    group_by(.data$cluster)
  return(crit_df)
}

.crit_proc <- function(df, limit) {
  arrange(df, .data$cluster, desc(.data$error)) %>%
    slice(1:limit) %>%
    mutate(rank = row_number()) %>%
    select(.data$cluster, .data$rank, everything())
}

.plot_topsols_errors <- function(df, top_sols, limit = 1, balance = rep(1, 3)) {
  balance <- balance / sum(balance)
  left_join(df, select(top_sols, 1:3), "solID") %>%
    mutate(
      alpha = ifelse(is.na(.data$cluster), 0.5, 1),
      label = ifelse(!is.na(.data$cluster), sprintf(
        "[%s.%s]", .data$cluster, .data$rank
      ), NA)
    ) %>%
    ggplot(aes(x = .data$nrmse, y = .data$decomp.rssd)) +
    geom_point(aes(colour = .data$cluster, alpha = .data$alpha)) +
    geom_text(aes(label = .data$label), na.rm = TRUE, hjust = -0.3) +
    guides(alpha = "none", colour = "none") +
    labs(
      title = paste("Selecting Top", limit, "Performing Models by Cluster"),
      subtitle = "Based on minimum (weighted) distance to origin",
      x = "NRMSE", y = "DECOMP.RSSD",
      caption = sprintf(
        "Weights: NRMSE %s%%, DECOMP.RSSD %s%%, MAPE %s%%",
        round(100 * balance[1]), round(100 * balance[2]), round(100 * balance[3])
      )
    ) +
    theme_lares()
}

.plot_topsols_rois <- function(top_sols, all_media, limit = 1) {
  top_sols %>%
    mutate(label = sprintf("[%s.%s]\n%s", .data$cluster, .data$rank, .data$solID)) %>%
    tidyr::gather("media", "roi", contains(all_media)) %>%
    ggplot(aes(x = .data$media, y = .data$roi)) +
    facet_grid(.data$label ~ .) +
    geom_col() +
    coord_flip() +
    labs(
      title = paste("ROIs on Top", limit, "Performing Models by Cluster"),
      x = NULL, y = "ROI per Media"
    ) +
    theme_lares()
}
