# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Clustering to Reduce Number of Models based on ROI and Errors
#'
#' \code{robyn_clusters()} uses output from \code{robyn_run()},
#' to reduce the number of models and help the user pick up the best (lowest
#' combined error) of the most different kinds (clusters) of models.
#'
#' @inheritParams lares::clusterKmeans
#' @inheritParams hyper_names
#' @inheritParams robyn_outputs
#' @param input \code{robyn_export()}'s output or \code{pareto_aggregated.csv} results.
#' @param limit Integer. Top N results per cluster. If kept in "auto", will select k
#' as the cluster in which the WSS variance was less than 5\%.
#' @param weights Vector, size 3. How much should each error weight?
#' Order: nrmse, decomp.rssd, mape. The highest the value, the closer it will be scaled
#' to origin. Each value will be normalized so they all sum 1.
#' @param export Export plots into local files?
#' @param ... Additional parameters passed to \code{lares::clusterKmeans()}.
#' @author Bernardo Lares (bernardolares@@fb.com)
#' @examples
#' \dontrun{
#' # Having InputCollect and OutputCollect results
#' cls <- robyn_clusters(
#'   input = OutputCollect,
#'   all_media = InputCollect$all_media,
#'   k = 3, limit = 2,
#'   weights = c(1, 1, 1.5)
#' )
#' }
#' @return List. Clustering results as labeled data.frames and plots.
#' @export
robyn_clusters <- function(input, all_media = NULL, k = "auto", limit = 1,
                           weights = rep(1, 3), dim_red = "PCA",
                           quiet = FALSE, export = FALSE,
                           ...) {
  if ("robyn_outputs" %in% class(input)) {
    if (is.null(all_media)) {
      aux <- colnames(input$mediaVecCollect)
      all_media <- aux[-c(1, which(aux == "type"):length(aux))]
      path <- input$plot_folder
    } else {
      path <- paste0(getwd(), "/")
    }
    # Pareto and ROI data
    xDecompAgg <- input$xDecompAgg
    df <- .prepare_df(xDecompAgg, all_media = all_media)
  } else {
    if (all(c("solID", "mape", "nrmse", "decomp.rssd") %in% names(input)) & is.data.frame(input)) {
      df <- .prepare_df(input, all_media)
    } else {
      stop(paste(
        "You must run robyn_outputs(..., clusters = TRUE) or",
        "pass a valid data.frame (sames as pareto_aggregated.csv output)",
        "in order to use robyn_clusters()"
      ))
    }
  }

  ignore <- c("solID", "mape", "decomp.rssd", "nrmse", "pareto")

  # Auto K selected by less than 5% WSS variance (convergence)
  min_clusters <- 3
  limit_clusters <- min(nrow(df) - 1, 30)
  if ("auto" %in% k) {
    cls <- tryCatch(
      {
        clusterKmeans(df, k = NULL, limit = limit_clusters, ignore = ignore, dim_red = dim_red, quiet = TRUE, ...)
      },
      error = function(err) {
        message(paste("Couldn't automatically create clusters:", err))
        return(NULL)
      }
    )
    # if (is.null(cls)) return(NULL)
    min_var <- 0.05
    k <- cls$nclusters %>%
      mutate(
        pareto = .data$wss / .data$wss[1],
        dif = lag(.data$pareto) - .data$pareto
      ) %>%
      filter(.data$dif > min_var) %>%
      pull(.data$n) %>%
      max(.)
    if (k < min_clusters) k <- min_clusters
    if (!quiet) {
      message(sprintf(
        ">> Auto selected k = %s (clusters) based on minimum WSS variance of %s%%",
        k, min_var * 100
      ))
    }
  }

  # Build clusters
  stopifnot(k %in% min_clusters:30)
  cls <- clusterKmeans(df, k, limit = limit_clusters, ignore = ignore, dim_red = dim_red, quiet = TRUE, ...)

  # Select top models by minimum (weighted) distance to zero
  top_sols <- .clusters_df(cls$df, weights, limit)

  output <- list(
    # Data and parameters
    data = mutate(cls$df, top_sol = .data$solID %in% top_sols$solID),
    n_clusters = k,
    errors_weights = weights,
    # Within Groups Sum of Squares Plot
    wss = cls$nclusters_plot,
    # Grouped correlations per cluster
    corrs = cls$correlations + labs(title = "Top Correlations by Cluster", subtitle = NULL),
    # Mean ROI per cluster
    clusters_means = cls$means,
    # Dim reduction clusters
    clusters_PCA = cls[["PCA"]],
    clusters_tSNE = cls[["tSNE"]],
    # Top Clusters
    models = top_sols,
    plot_models_errors = .plot_topsols_errors(df, top_sols, limit, weights),
    plot_models_rois = .plot_topsols_rois(df, top_sols, all_media, limit)
  )

  if (export) {
    write.csv(output$data, file = paste0(path, "pareto_clusters.csv"))
    ggsave(paste0(path, "pareto_clusters_wss.png"), plot = output$wss, dpi = 500, width = 5, height = 4)
    # ggsave(paste0(path, "pareto_clusters_corr.png"), plot = output$corrs, dpi = 500, width = 7, height = 5)
    db <- wrap_plots(output$plot_models_rois, output$plot_models_errors)
    ggsave(paste0(path, "pareto_clusters_detail.png"), plot = db, dpi = 600, width = 12, height = 9)
  }

  return(output)
}

errors_scores <- function(df, balance = rep(1, 3)) {
  stopifnot(length(balance) == 3)
  error_cols <- c("nrmse", "decomp.rssd", "mape")
  stopifnot(all(error_cols %in% colnames(df)))
  balance <- balance / sum(balance)
  scores <- df %>%
    select(all_of(error_cols)) %>%
    # Force normalized values so they can be comparable
    mutate(
      nrmse_n = .min_max_norm(.data$nrmse),
      decomp.rssd_n = .min_max_norm(.data$decomp.rssd),
      mape_n = .min_max_norm(.data$mape)
    ) %>%
    replace(., is.na(.), 0) %>%
    # Balance to give more or less importance to each error
    mutate(
      nrmse_w = balance[1] * .data$nrmse_n,
      decomp.rssd_w = balance[2] * .data$decomp.rssd_n,
      mape_w = balance[3] * .data$mape_n
    ) %>%
    # Calculate error score
    mutate(error_score = (.data$nrmse_w^2 + .data$decomp.rssd_w^2 + .data$mape_w^2)^-(1 / 2)) %>%
    pull(.data$error_score)
  return(scores)
}

# # Mean Media ROI by Cluster
# df %>%
#   mutate(cluster = sprintf("Cluster %s", cls$df$cluster)) %>%
#   select(-.data$mape, -.data$decomp.rssd, -.data$nrmse, -.data$solID) %>%
#   group_by(.data$cluster) %>%
#   summarize_all(list(mean)) %>%
#   tidyr::pivot_longer(-one_of("cluster"), names_to = "media", values_to = "meanROI") %>%
#   ggplot(aes(y = reorder(.data$media, .data$meanROI), x = .data$meanROI)) +
#   facet_grid(.data$cluster~.) +
#   geom_col() + theme_lares() +
#   labs(title = "Mean Media ROI by Cluster",
#        x = "(Un-normalized) mean ROI within clsuter", y = NULL)
# df %>%
#   mutate(cluster = sprintf("Cluster %s", cls$df$cluster)) %>%
#   select(-.data$solID, -.data$mape, -.data$decomp.rssd, -.data$nrmse) %>%
#   tidyr::pivot_longer(-one_of("cluster"), names_to = "media", values_to = "roi") %>%
#   ggplot(aes(y = reorder(.data$media, .data$roi), x = .data$roi)) +
#   facet_grid(.data$cluster~.) +
#   geom_boxplot() + theme_lares() +
#   labs(title = "Media ROI by Cluster",
#        x = "(Un-normalized) ROI", y = NULL)

# ROIs data.frame for clustering (from xDecompAgg or pareto_aggregated.csv)
.prepare_df <- function(x, all_media) {
  check_opts(all_media, unique(x$rn))
  rois <- select(x, .data$solID, .data$rn, .data$roi_total) %>%
    tidyr::spread(key = .data$rn, value = .data$roi_total)
  rois <- removenacols(rois, all = FALSE)
  rois <- select(rois, any_of(c("solID", all_media)))
  errors <- distinct(x, .data$solID, .data$nrmse, .data$decomp.rssd, .data$mape)
  rois <- left_join(rois, errors, "solID") %>% ungroup()
  return(rois)
}

.min_max_norm <- function(x, min = 0, max = 1) {
  if (length(x) == 1) {
    return(x)
  } # return((max - min) / 2)
  a <- min(x, na.rm = TRUE)
  b <- max(x, na.rm = TRUE)
  (max - min) * (x - a) / (b - a) + min
}

.clusters_df <- function(df, balance = rep(1, 3), limit = 1) {
  df %>%
    mutate(error_score = errors_scores(., balance)) %>%
    replace(., is.na(.), 0) %>%
    group_by(.data$cluster) %>%
    arrange(.data$cluster, .data$solID, desc(.data$error_score)) %>%
    slice(1:limit) %>%
    mutate(rank = row_number()) %>%
    select(.data$cluster, .data$rank, everything())
}

.plot_topsols_errors <- function(df, top_sols, limit = 1, balance = rep(1, 3)) {
  balance <- balance / sum(balance)
  left_join(df, select(top_sols, 1:3), "solID") %>%
    mutate(
      alpha = ifelse(is.na(.data$cluster), 0.6, 1),
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

.plot_topsols_rois <- function(df, top_sols, all_media, limit = 1) {
  real_rois <- as.data.frame(df)[, -c(which(colnames(df) %in% c("mape", "nrmse", "decomp.rssd")))]
  colnames(real_rois) <- paste0("real_", colnames(real_rois))
  top_sols %>%
    left_join(real_rois, by = c("solID" = "real_solID")) %>%
    mutate(label = sprintf("[%s.%s]\n%s", .data$cluster, .data$rank, .data$solID)) %>%
    tidyr::gather("media", "roi", contains(all_media)) %>%
    filter(grepl("real_", .data$media)) %>%
    mutate(media = gsub("real_", "", .data$media)) %>%
    ggplot(aes(x = reorder(.data$media, .data$roi), y = .data$roi)) +
    facet_grid(.data$label ~ .) +
    geom_col() +
    coord_flip() +
    labs(
      title = paste("Top Performing Models"),
      x = NULL, y = "Mean metric per media"
    ) +
    theme_lares()
}
