import numpy as np
import pandas as pd
import warnings

from .transformation import saturation_hill, transform_adstock


def robyn_calibrate(
    calibration_input,
    df_raw,
    dayInterval,
    xDecompVec,
    coefs,
    hypParamSam,
    wind_start,
    wind_end,
    adstock,
):
    # Convert the R code's dataframe to a Pandas dataframe
    df_raw = pd.DataFrame(df_raw)

    # Extract the necessary columns from the dataframe
    ds_wind = df_raw["ds"][(wind_start-1):(wind_end-1)]
    min_ds = calibration_input["liftStartDate"] >= np.min(ds_wind)
    max_ds = calibration_input["liftEndDate"] <= (np.max(ds_wind) - np.timedelta64(dayInterval - 1,'D'))
    include_study = np.any(min_ds & max_ds)

    if not include_study:
        warnings.warn(
            "All calibration_input in outside modelling window. Running without calibration"
        )  ## Manually added
        return None
    elif include_study and calibration_input is not None:

        calibration_input["pred"] = pd.NA
        calibration_input["pred_total"] = pd.NA
        calibration_input["decompStart"] = pd.NA
        calibration_input["decompEnd"] = pd.NA

        # Split the channels into a list
        temp_channels = calibration_input["channel"].values
        split_channels = list()
        for val in temp_channels:
            str_list = str(val).split("+")
            split_channels.append(str_list)
            # for channel_str in str_list:

        ##split_channels = list(calibration_input["channel"].values)

        # Initialize the list of calibrated channels
        l_chn_collect = []
        l_chn_total_collect = []

        # Loop through each channel
        for l_study in range(len(split_channels)):
            # Get the current channel and its corresponding scope
            get_channels = split_channels[l_study]
            scope = calibration_input["calibration_scope"].values[l_study]
            study_start = calibration_input["liftStartDate"].values[l_study]
            study_end = calibration_input["liftEndDate"].values[l_study]
            study_pos = df_raw.index[(df_raw['ds'] >= study_start) & (df_raw['ds'] <= study_end)].values

            if study_start in df_raw['ds'].values:
                calib_pos = study_pos
            else:
                calib_pos = [min(study_pos) - 1]
                calib_pos.extend(study_pos)
                ##calib_pos = min_list

            calibrate_dates = df_raw.loc[calib_pos, "ds"].values
            ##TODO: xDecompVec input is wrong, possibly a join error done in model.py
            calib_pos_rw = xDecompVec.index[calibrate_dates in xDecompVec['ds']].values

            # Initialize the list of calibrated channels for this scope
            l_chn_collect_scope = []
            l_chn_total_collect_scope = []

            # Loop through each position in the channel
            for l_chn in range(len(get_channels)):
                # Get the current channel and its corresponding position
                if scope == 'immediate':
                    channel = get_channels[l_chn]
                    m = df_raw[get_channels[l_chn]].values[0]
                    pos = np.where(df_raw["ds"] == channel)[0]

                    if adstock == 'geometric':
                        theta = hypParamSam[get_channels[l_chn] + "_thetas"].values[0][0]

                    if adstock.startswith('weibull'):
                        shape = hypParamSam[get_channels[l_chn] + "_shapes"].values[0][0]
                        scale = hypParamSam[get_channels[l_chn] + "_scales"].values[0][0]

                    x_list = transform_adstock(m, adstock, theta=theta, shape=shape, scale=scale)

                    if adstock == "weibull_pdf":
                        m_imme = x_list['x_imme']
                    else:
                        m_imme = m

                    m_total = x_list['x_decayed']
                    m_coav = m_total - m_imme

                    m_caov_calib = m_caov[calib_pos]
                    m_total_rw = m_total[wind_start:wind_end]

                    alpha = hypParamSam[get_channels[l_chn] + "_alphas"].values[0][0]
                    gamma = hypParamSam[get_channels[l_chn] + "_gammas"].values[0][0]

                    m_calib_caov_sat = saturation_hill(m_total_rw, alpha = alpha, gamma = gamma, x_marginal = m_caov_calib)

                    m_calib_caov_decomp = m_calib_caov_sat * coefs['s0'][coefs['rn'] == get_channels[l_chn]]
                    m_calib_total_decomp = xDecompVec[calib_pos_rw, get_channels[l_chn]]

                    m_calib_decomp = m_calib_total_decomp - m_calib_caov_decomp

                    if scope == 'total':
                        m_calib_decomp = m_calib_total_decomp = xDecompVec[calib_pos_rw, get_channels[l_chn]]

                    l_chn_collect[[get_channels[l_chn]]] = m_calib_decomp
                    l_chn_total_collect[[get_channels[l_chn]]] = m_calib_total_decomp

            if len(get_channels) > 1:
                l_chn_collect = l_chn_collect.sum(axis=0) ##rowSums[bind_cols(l_chn_collect))
                l_chn_total_collect = l_chn_total_collect.sum(axis=0) ##rowSums(bind_cols(l_chn_total_collect))
            else:
                l_chn_collect = l_chn_collect ##unlist(l_chn_collect, use.names = FALSE)
                l_chn_total_collect = l_chn_total_collect ##unlist(l_chn_total_collect, use.names = FALSE)

            # Check if the position is within the modelling window
            ## if pos.size > 0:
            ##    # Extract the necessary data for this position
            ##    study_start = calibration_input["liftStartDate"][l_study]
            ##    study_end = calibration_input["liftEndDate"][l_study]
            ##    study_pos = np.where(
            ##        df_raw["ds"] >= study_start & df_raw["ds"] <= study_end
            ##    )[0]

                # Calculate the calibrated values for this position
            ##    calib_pos = study_pos - pos[0] + 1
            ##    calib_dates = df_raw.iloc[calib_pos, 0]

                # Calculate the calibrated values for this channel
            ##    calib_chn = np.sum(xDecompVec[calib_pos, channel])

                # Add the calibrated values to the list
            ##    l_chn_collect_scope.append(calib_chn)
            ##    l_chn_total_collect_scope.append(calib_chn)

            calibration_input[l_study]


        # Combine the lists for this scope
        l_chn_collect.append(np.sum(l_chn_collect_scope))
        l_chn_total_collect.append(np.sum(l_chn_total_collect_scope))

        # Combine the lists for all scopes
        l_chn_collect = np.sum(l_chn_collect, axis=0)
        l_chn_total_collect = np.sum(l_chn_total_collect, axis=0)

        # Calculate the calibrated values for the total channel
        l_chn_total_collect = np.sum(xDecompVec[wind_start:wind_end, :])

        # Calculate the calibrated values for the total channel
        l_chn_collect = np.sum(l_chn_collect, axis=0)

        # Calculate the MAPE for the total channel
        mape_lift = np.abs((l_chn_collect - l_chn_total_collect) / l_chn_total_collect)

        # Create a new dataframe with the calibrated values
        liftCollect = pd.DataFrame(
            {
                "liftMedia": df_raw["channel"],
                "liftStart": df_raw["liftStartDate"],
                "liftEnd": df_raw["liftEndDate"],
                "liftAbs": l_chn_total_collect,
                "decompStart": np.min(ds_wind),
                "decompEnd": np.max(ds_wind) + dayInterval - 1,
                "decompAbsScaled": l_chn_collect,
                "decompAbsTotalScaled": l_chn_total_collect,
                "calibrated_pct": l_chn_collect / l_chn_total_collect,
                "mape_lift": mape_lift,
            }
        )

        return liftCollect
