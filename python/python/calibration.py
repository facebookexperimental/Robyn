import warnings

import numpy as np
import pandas as pd


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
    ds_wind = df_raw["ds"][wind_start:wind_end]
    include_study = np.any(
        calibration_input["liftStartDate"]
        >= np.min(ds_wind) & calibration_input["liftEndDate"]
        <= np.max(ds_wind) + dayInterval - 1
    )

    if not include_study:
        warnings.warn(
            "All calibration_input in outside modelling window. Running without calibration"
        )  ## Manually added
        return None

    # Split the channels into a list
    split_channels = df_raw["channel"].str.split(r"\+")

    # Initialize the list of calibrated channels
    l_chn_collect = []
    l_chn_total_collect = []

    # Loop through each channel
    for l_study in range(len(split_channels)):
        # Get the current channel and its corresponding scope
        get_channels = split_channels[l_study]
        scope = calibration_input["calibration_scope"][l_study]

        # Initialize the list of calibrated channels for this scope
        l_chn_collect_scope = []
        l_chn_total_collect_scope = []

        # Loop through each position in the channel
        for l_chn in range(len(get_channels)):
            # Get the current channel and its corresponding position
            channel = get_channels[l_chn]
            pos = np.where(df_raw["ds"] == channel)[0]

            # Check if the position is within the modelling window
            if pos.size > 0:
                # Extract the necessary data for this position
                study_start = calibration_input["liftStartDate"][l_study]
                study_end = calibration_input["liftEndDate"][l_study]
                study_pos = np.where(
                    df_raw["ds"] >= study_start & df_raw["ds"] <= study_end
                )[0]

                # Calculate the calibrated values for this position
                calib_pos = study_pos - pos[0] + 1
                calib_dates = df_raw.iloc[calib_pos, 0]

                # Calculate the calibrated values for this channel
                calib_chn = np.sum(xDecompVec[calib_pos, channel])

                # Add the calibrated values to the list
                l_chn_collect_scope.append(calib_chn)
                l_chn_total_collect_scope.append(calib_chn)

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
