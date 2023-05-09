# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

robyn_calibrate <- function(calibration_input,
                            df_raw, # df_raw = InputCollect$dt_mod
                            dayInterval, # dayInterval = InputCollect$dayInterval
                            xDecompVec, # xDecompVec = decompCollect$xDecompVec
                            coefs, # coefs = decompCollect$coefsOutCat
                            hypParamSam,
                            wind_start = 1,
                            wind_end = nrow(df_raw),
                            adstock) {
  ds_wind <- df_raw$ds[wind_start:wind_end]
  include_study <- any(
    calibration_input$liftStartDate >= min(ds_wind) &
      calibration_input$liftEndDate <= (max(ds_wind) + dayInterval - 1)
  )
  if (!is.null(calibration_input) & !include_study) {
    warning("All calibration_input in outside modelling window. Running without calibration")
  } else if (!is.null(calibration_input) & include_study) {
    calibration_input <- mutate(
      calibration_input,
      pred = NA, pred_total = NA, decompStart = NA, decompEnd = NA
    )
    split_channels <- strsplit(calibration_input$channel, split = "\\+")

    for (l_study in seq_along(split_channels)) {
      get_channels <- split_channels[[l_study]]
      scope <- calibration_input$calibration_scope[[l_study]]
      study_start <- calibration_input$liftStartDate[[l_study]]
      study_end <- calibration_input$liftEndDate[[l_study]]
      study_pos <- which(df_raw$ds >= study_start & df_raw$ds <= study_end)
      if (study_start %in% df_raw$ds) {
        calib_pos <- study_pos
      } else {
        calib_pos <- c(min(study_pos) - 1, study_pos)
      }
      calibrate_dates <- df_raw[calib_pos, "ds"][[1]]
      calib_pos_rw <- which(xDecompVec$ds %in% calibrate_dates)

      l_chn_collect <- list()
      l_chn_total_collect <- list()
      for (l_chn in seq_along(get_channels)) { # l_chn =1
        if (scope == "immediate") {
          m <- df_raw[, get_channels[l_chn]][[1]]
          # m_calib <- df_raw[calib_pos, get_channels[l_chn]][[1]]

          ## 1. Adstock
          if (adstock == "geometric") {
            theta <- hypParamSam[paste0(get_channels[l_chn], "_thetas")][[1]][[1]]
          }
          if (grepl("weibull", adstock)) {
            shape <- hypParamSam[paste0(get_channels[l_chn], "_shapes")][[1]][[1]]
            scale <- hypParamSam[paste0(get_channels[l_chn], "_scales")][[1]][[1]]
          }
          x_list <- transform_adstock(m, adstock, theta = theta, shape = shape, scale = scale)
          if (adstock == "weibull_pdf") {
            m_imme <- x_list$x_imme
          } else {
            m_imme <- m
          }
          m_total <- x_list$x_decayed
          m_caov <- m_total - m_imme

          ## 2. Saturation
          m_caov_calib <- m_caov[calib_pos]
          m_total_rw <- m_total[wind_start:wind_end]
          alpha <- hypParamSam[paste0(get_channels[l_chn], "_alphas")][[1]][[1]]
          gamma <- hypParamSam[paste0(get_channels[l_chn], "_gammas")][[1]][[1]]
          m_calib_caov_sat <- saturation_hill(
            m_total_rw,
            alpha = alpha, gamma = gamma, x_marginal = m_caov_calib
          )
          m_calib_caov_decomp <- m_calib_caov_sat * coefs$s0[coefs$rn == get_channels[l_chn]]
          m_calib_total_decomp <- xDecompVec[calib_pos_rw, get_channels[l_chn]]
          m_calib_decomp <- m_calib_total_decomp - m_calib_caov_decomp
        }
        if (scope == "total") {
          m_calib_decomp <- m_calib_total_decomp <- xDecompVec[calib_pos_rw, get_channels[l_chn]]
        }
        l_chn_collect[[get_channels[l_chn]]] <- m_calib_decomp
        l_chn_total_collect[[get_channels[l_chn]]] <- m_calib_total_decomp
      }

      if (length(get_channels) > 1) {
        l_chn_collect <- rowSums(bind_cols(l_chn_collect))
        l_chn_total_collect <- rowSums(bind_cols(l_chn_total_collect))
      } else {
        l_chn_collect <- unlist(l_chn_collect, use.names = FALSE)
        l_chn_total_collect <- unlist(l_chn_total_collect, use.names = FALSE)
      }

      calibration_input[l_study, ] <- mutate(
        calibration_input[l_study, ],
        pred = sum(l_chn_collect),
        pred_total = sum(l_chn_total_collect),
        decompStart = range(calibrate_dates)[1],
        decompEnd = range(calibrate_dates)[2]
      )
    }
    liftCollect <- calibration_input %>%
      mutate(
        decompStart = as.Date(.data$decompStart, "1970-01-01"),
        decompEnd = as.Date(.data$decompEnd, "1970-01-01")
      ) %>%
      mutate(
        liftDays = as.numeric(
          difftime(.data$liftEndDate, .data$liftStartDate, units = "days")
        ),
        decompDays = as.numeric(
          difftime(.data$decompEnd, .data$decompStart, units = "days")
        )
      ) %>%
      mutate(
        decompAbsScaled = .data$pred / .data$decompDays * .data$liftDays,
        decompAbsTotalScaled = .data$pred_total / .data$decompDays * .data$liftDays
      ) %>%
      mutate(
        liftMedia = .data$channel,
        liftStart = .data$liftStartDate,
        liftEnd = .data$liftEndDate,
        mape_lift = abs((.data$decompAbsScaled - .data$liftAbs) / .data$liftAbs),
        calibrated_pct = .data$decompAbsScaled / .data$decompAbsTotalScaled
      ) %>%
      dplyr::select(
        .data$liftMedia, .data$liftStart, .data$liftEnd, .data$liftAbs,
        .data$decompStart, .data$decompEnd, .data$decompAbsScaled,
        .data$decompAbsTotalScaled, .data$calibrated_pct, .data$mape_lift
      )

    return(liftCollect)
  }
}
