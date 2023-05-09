# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Clustering to Reduce Number of Models based on ROI and Errors
#'
#' \code{robyn_clusters()} uses output from \code{robyn_run()},
#' to reduce the number of models and create bootstrapped confidence
#' interval and help the user pick up the best (lowest combined error)
#' of the most different kinds (clusters) of models.
#'
#' @inheritParams lares::clusterKmeans
#' @inheritParams hyper_names
#' @inheritParams robyn_outputs
#' @param input \code{robyn_export()}'s output or \code{pareto_aggregated.csv} results.
#' @param dep_var_type Character. For dep_var_type 'revenue', ROI is used for clustering.
#' For conversion', CPA is used for clustering.
#' @param limit Integer. Top N results per cluster. If kept in "auto", will select k
#' as the cluster in which the WSS variance was less than 5\%.
#' @param weights Vector, size 3. How much should each error weight?
#' Order: nrmse, decomp.rssd, mape. The highest the value, the closer it will be scaled
#' to origin. Each value will be normalized so they all sum 1.
#' @param export Export plots into local files?
#' @param ... Additional parameters passed to \code{lares::clusterKmeans()}.
#' @author Bernardo Lares (bernardolares@@meta.com)
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
robyn_clusters <- function(input, dep_var_type, all_media = NULL, k = "auto", limit = 1,
                           weights = rep(1, 3), dim_red = "PCA",
                           quiet = FALSE, export = FALSE, seed = 123,
                           ...) {
  set.seed(seed)
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
    df <- .prepare_df(xDecompAgg, all_media, dep_var_type)
  } else {
    if (all(c("solID", "mape", "nrmse", "decomp.rssd") %in% names(input)) && is.data.frame(input)) {
      df <- .prepare_df(input, all_media, dep_var_type)
    } else {
      stop(paste(
        "You must run robyn_outputs(..., clusters = TRUE) or",
        "pass a valid data.frame (sames as pareto_aggregated.csv output)",
        "in order to use robyn_clusters()"
      ))
    }
  }

  ignore <- c("solID", "mape", "decomp.rssd", "nrmse", "nrmse_test", "nrmse_train", "pareto")

  # Auto K selected by less than 5% WSS variance (convergence)
  min_clusters <- 3
  limit_clusters <- min(nrow(df) - 1, 30)
  if ("auto" %in% k) {
    cls <- tryCatch(
      {
        clusterKmeans(df,
          k = NULL, limit = limit_clusters, ignore = ignore,
          dim_red = dim_red, quiet = TRUE, seed = seed
        ) # , ...)
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
      max(., na.rm = TRUE)
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
  cls <- clusterKmeans(df, k, limit = limit_clusters, ignore = ignore, dim_red = dim_red, quiet = TRUE) # , ...)

  # Select top models by minimum (weighted) distance to zero
  all_paid <- setdiff(names(cls$df), c(ignore, "cluster"))
  ts_validation <- ifelse(all(is.na(cls$df$nrmse_test)), FALSE, TRUE)
  top_sols <- .clusters_df(df = cls$df, all_paid, balance = weights, limit, ts_validation)

  # Build in-cluster CI with bootstrap
  ci_list <- confidence_calcs(xDecompAgg, cls, all_paid, dep_var_type, k, ...)

  output <- list(
    # Data and parameters
    data = mutate(cls$df, top_sol = .data$solID %in% top_sols$solID, cluster = as.integer(.data$cluster)),
    df_cluster_ci = ungroup(ci_list$df_ci) %>% dplyr::select(-.data$cluster_title),
    n_clusters = k,
    boot_n = ci_list$boot_n,
    sim_n = ci_list$sim_n,
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
    plot_clusters_ci = .plot_clusters_ci(ci_list$sim_collect, ci_list$df_ci, dep_var_type, ci_list$boot_n, ci_list$sim_n),
    plot_models_errors = .plot_topsols_errors(df, top_sols, limit, weights),
    plot_models_rois = .plot_topsols_rois(df, top_sols, all_media, limit)
  )

  if (export) {
    write.csv(output$data, file = paste0(path, "pareto_clusters.csv"))
    write.csv(output$df_cluster_ci, file = paste0(path, "pareto_clusters_ci.csv"))
    ggsave(paste0(path, "pareto_clusters_wss.png"), plot = output$wss, dpi = 500, width = 5, height = 4)
    db <- wrap_plots(
      A = output$plot_clusters_ci,
      B = output$plot_models_rois,
      C = output$plot_models_errors,
      design = "AA\nBC"
    )
    # Suppressing "Picking joint bandwidth of x" messages
    suppressMessages(ggsave(paste0(path, "pareto_clusters_detail.png"),
      plot = db, dpi = 500, width = 12, height = 14
    ))
  }

  return(output)
}

confidence_calcs <- function(xDecompAgg, cls, all_paid, dep_var_type, k, boot_n = 1000, sim_n = 10000, ...) {
  df_clusters_outcome <- xDecompAgg %>%
    filter(!is.na(.data$total_spend)) %>%
    left_join(y = dplyr::select(cls$df, c("solID", "cluster")), by = "solID") %>%
    dplyr::select(c("solID", "cluster", "rn", "roi_total", "cpa_total", "robynPareto")) %>%
    group_by(.data$cluster, .data$rn) %>%
    mutate(n = n()) %>%
    filter(!is.na(.data$cluster)) %>%
    arrange(.data$cluster, .data$rn)

  cluster_collect <- list()
  chn_collect <- list()
  sim_collect <- list()
  for (j in 1:k) {
    df_outcome <- filter(df_clusters_outcome, .data$cluster == j)
    if (length(unique(df_outcome$solID)) < 3) {
      warning(paste("Cluster", j, "does not contain enough models to calculate CI"))
    } else {
      for (i in all_paid) {
        # Bootstrap CI
        if (dep_var_type == "conversion") {
          # Drop CPA == Inf
          df_chn <- filter(df_outcome, .data$rn == i & is.finite(.data$cpa_total))
          v_samp <- df_chn$cpa_total
        } else {
          df_chn <- filter(df_outcome, .data$rn == i)
          v_samp <- df_chn$roi_total
        }
        boot_res <- .bootci(samp = v_samp, boot_n = boot_n)
        boot_mean <- mean(boot_res$boot_means)
        boot_se <- boot_res$se
        ci_low <- ifelse(boot_res$ci[1] < 0, 0, boot_res$ci[1])
        ci_up <- boot_res$ci[2]

        # Collect loop results
        chn_collect[[i]] <- df_chn %>%
          mutate(
            ci_low = ci_low,
            ci_up = ci_up,
            n = length(v_samp),
            boot_se = boot_se,
            boot_mean = boot_mean,
            cluster = j
          )
        sim_collect[[i]] <- data.frame(
          cluster = j,
          rn = i,
          n = length(v_samp),
          boot_mean = boot_mean,
          x_sim = rnorm(sim_n, mean = boot_mean, sd = boot_se)
        ) %>%
          mutate(y_sim = dnorm(.data$x_sim, mean = boot_mean, sd = boot_se))
      }
    }
    cluster_collect[[j]] <- list(chn_collect = chn_collect, sim_collect = sim_collect)
  }

  sim_collect <- bind_rows(lapply(cluster_collect, function(x) {
    bind_rows(lapply(x$sim_collect, function(y) y))
  })) %>%
    mutate(cluster_title = sprintf("Cl.%s (n=%s)", .data$cluster, .data$n)) %>%
    ungroup() %>%
    as_tibble()

  df_ci <- bind_rows(lapply(cluster_collect, function(x) {
    bind_rows(lapply(x$chn_collect, function(y) y))
  })) %>%
    mutate(cluster_title = sprintf("Cl.%s (n=%s)", .data$cluster, .data$n)) %>%
    dplyr::select(
      .data$rn, .data$cluster_title, .data$n, .data$cluster,
      .data$boot_mean, .data$boot_se, .data$ci_low, .data$ci_up
    ) %>%
    distinct() %>%
    group_by(.data$rn, .data$cluster_title, .data$cluster) %>%
    summarise(
      n = .data$n,
      boot_mean = .data$boot_mean,
      boot_se = boot_se,
      boot_ci = sprintf("[%s, %s]", round(.data$ci_low, 2), round(.data$ci_up, 2)),
      ci_low = .data$ci_low,
      ci_up = .data$ci_up,
      sd = boot_se * sqrt(.data$n - 1),
      dist100 = (.data$ci_up - .data$ci_low + 2 * boot_se * sqrt(.data$n - 1)) / 99,
      .groups = "drop"
    ) %>%
    ungroup()
  return(list(
    df_ci = df_ci,
    sim_collect = sim_collect,
    boot_n = boot_n,
    sim_n = sim_n
  ))
}

errors_scores <- function(df, balance = rep(1, 3), ts_validation = TRUE, ...) {
  stopifnot(length(balance) == 3)
  error_cols <- c(ifelse(ts_validation, "nrmse_test", "nrmse_train"), "decomp.rssd", "mape")
  stopifnot(all(error_cols %in% colnames(df)))
  balance <- balance / sum(balance)
  scores <- df %>%
    select(all_of(error_cols)) %>%
    rename("nrmse" = 1) %>%
    mutate(
      nrmse = ifelse(is.infinite(.data$nrmse), max(is.finite(.data$nrmse)), .data$nrmse),
      decomp.rssd = ifelse(is.infinite(.data$decomp.rssd), max(is.finite(.data$decomp.rssd)), .data$decomp.rssd),
      mape = ifelse(is.infinite(.data$mape), max(is.finite(.data$mape)), .data$mape)
    ) %>%
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
    mutate(error_score = sqrt(.data$nrmse_w^2 + .data$decomp.rssd_w^2 + .data$mape_w^2)) %>%
    pull(.data$error_score)
  return(scores)
}

# ROIs data.frame for clustering (from xDecompAgg or pareto_aggregated.csv)
.prepare_df <- function(x, all_media, dep_var_type) {
  check_opts(all_media, unique(x$rn))

  if (dep_var_type == "revenue") {
    outcome <- select(x, .data$solID, .data$rn, .data$roi_total) %>%
      tidyr::spread(key = .data$rn, value = .data$roi_total)
  } else {
    outcome <- select(x, .data$solID, .data$rn, .data$cpa_total) %>%
      filter(is.finite(.data$cpa_total)) %>%
      tidyr::spread(key = .data$rn, value = .data$cpa_total)
  }

  outcome <- removenacols(outcome, all = FALSE)
  outcome <- select(outcome, any_of(c("solID", all_media)))
  errors <- distinct(
    x, .data$solID, .data$nrmse, .data$nrmse_test,
    .data$nrmse_train, .data$decomp.rssd, .data$mape
  )
  outcome <- left_join(outcome, errors, "solID") %>% ungroup()
  return(outcome)
}

.min_max_norm <- function(x, min = 0, max = 1) {
  x <- x[is.finite(x)]
  if (length(x) == 1) {
    return(x)
  } # return((max - min) / 2)
  a <- min(x, na.rm = TRUE)
  b <- max(x, na.rm = TRUE)
  (max - min) * (x - a) / (b - a) + min
}

.clusters_df <- function(df, all_paid, balance = rep(1, 3), limit = 1, ts_validation = TRUE, ...) {
  df %>%
    mutate(error_score = errors_scores(., balance, ts_validation = ts_validation, ...)) %>%
    replace(., is.na(.), 0) %>%
    group_by(.data$cluster) %>%
    arrange(.data$cluster, .data$error_score) %>%
    slice(1:limit) %>%
    mutate(rank = row_number()) %>%
    select(.data$cluster, .data$rank, everything())
}

.plot_clusters_ci <- function(sim_collect, df_ci, dep_var_type, boot_n, sim_n) {
  temp <- ifelse(dep_var_type == "conversion", "CPA", "ROAS")
  df_ci <- df_ci[complete.cases(df_ci), ]
  p <- ggplot(sim_collect, aes(x = .data$x_sim, y = .data$rn)) +
    facet_wrap(~ .data$cluster_title) +
    geom_density_ridges_gradient(scale = 3, rel_min_height = 0.01, size = 0.1) +
    geom_text(
      data = df_ci,
      aes(x = .data$boot_mean, y = .data$rn, label = .data$boot_ci),
      position = position_nudge(x = -0.02, y = 0.1),
      colour = "grey30", size = 3.5
    ) +
    geom_vline(xintercept = 1, linetype = "dashed", size = .5, colour = "grey75") +
    # scale_fill_viridis_c(option = "D") +
    labs(
      title = paste("In-Cluster", temp, "& bootstrapped 95% CI"),
      subtitle = "Sampling distribution of cluster mean",
      x = temp,
      y = "Density",
      fill = temp,
      caption = sprintf(
        "Based on %s bootstrap results with %s simulations",
        formatNum(boot_n, abbr = TRUE),
        formatNum(sim_n, abbr = TRUE)
      )
    ) +
    theme_lares(legend = "none")
  if (temp == "ROAS") {
    p <- p + geom_hline(yintercept = 1, alpha = 0.5, colour = "grey50", linetype = "dashed")
  }
  return(p)
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

.bootci <- function(samp, boot_n, seed = 1, ...) {
  set.seed(seed)

  if (length(samp) > 1) {
    samp_n <- length(samp)
    samp_mean <- mean(samp, na.rm = TRUE)
    boot_sample <- matrix(
      sample(x = samp, size = samp_n * boot_n, replace = TRUE),
      nrow = boot_n, ncol = samp_n
    )
    boot_means <- apply(X = boot_sample, MARGIN = 1, FUN = mean)
    se <- sd(boot_means)
    # binwidth <- diff(range(boot_means))/30
    # plot_boot <- ggplot(data.frame(x = boot_means),aes(x = x)) +
    #   geom_histogram(aes(y = ..density.. ), binwidth = binwidth) +
    #   geom_density(color="red")
    me <- qt(0.975, samp_n - 1) * se
    # ci <- c(mean(boot_means) - me, mean(boot_means) + me)
    samp_me <- me * sqrt(samp_n)
    ci <- c(samp_mean - samp_me, samp_mean + samp_me)

    return(list(boot_means = boot_means, ci = ci, se = se))
  } else {
    return(list(boot_means = samp, ci = c(NA, NA), se = NA))
  }
}
