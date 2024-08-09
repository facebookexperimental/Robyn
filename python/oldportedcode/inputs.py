# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import numpy as np
import pandas as pd
import datetime as dt
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

import matplotlib
## Prevents plot windows showing up.
matplotlib.use('qtagg')

## from sklearn.model_selection import AIC, BIC ## Manual, there are no AIC or BIC functions in sklearn
## from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale

## Added lmfit to use nlsLM in Python.
## Manually added curve_fit for nlsLM fitting
## lmfit for nonlinear least squares fitting
from lmfit import create_params, fit_report, minimize
## Uses from scipy.optimize import curve_fit
## from scipy.stats import linregress
from lmfit.models import LinearModel

## Manually added ggplot
from plotnine import ggplot, aes, labs, geom_point, geom_line, theme_gray, scale_x_continuous, scale_y_continuous

## Manually added imports
##check_nas, check_varnames, check_datevar, check_depvar, check_prophet, check_context, check_paidmedia, check_organicvars, check_factorvars
from .checks import *



def robyn_inputs(
    dt_input=None,
    dep_var=None,
    dep_var_type=None,
    date_var="auto",
    paid_media_spends=None,
    paid_media_vars=None,
    paid_media_signs=None,
    organic_vars=None,
    organic_signs=None,
    context_vars=None,
    context_signs=None,
    factor_vars=None,
    dt_holidays=None,
    prophet_vars=None,
    prophet_signs=None,
    prophet_country=None,
    adstock=None,
    hyperparameters=None,
    window_start=None,
    window_end=None,
    calibration_input=None,
    json_file=None,
    InputCollect=None,
    **kwargs,
):
    """
    robyn_inputs function in Python.

    This function is used to collect and validate the input parameters for the Robyn model.

    Parameters:
    - dt_input: The input data as a pandas DataFrame.
    - dep_var: The dependent variable name.
    - dep_var_type: The type of the dependent variable (e.g., "continuous", "binary", "count").
    - date_var: The name of the date variable in the input data. Default is "auto".
    - paid_media_spends: The names of the paid media spend variables.
    - paid_media_vars: The names of the paid media variables.
    - paid_media_signs: The signs (+/-) for the paid media variables.
    - organic_vars: The names of the organic media variables.
    - organic_signs: The signs (+/-) for the organic media variables.
    - context_vars: The names of the context variables.
    - context_signs: The signs (+/-) for the context variables.
    - factor_vars: The names of the factor variables.
    - dt_holidays: The holidays data as a pandas DataFrame.
    - prophet_vars: The names of the variables to be used in the Prophet model.
    - prophet_signs: The signs (+/-) for the Prophet variables.
    - prophet_country: The country for the Prophet model.
    - adstock: The adstock parameters.
    - hyperparameters: The hyperparameters for the model.
    - window_start: The start date of the modeling window.
    - window_end: The end date of the modeling window.
    - calibration_input: The calibration input data as a pandas DataFrame.
    - json_file: The path to a JSON file containing the input parameters.
    - InputCollect: The collected input parameters.

    Returns:
    - InputCollect: The collected and validated input parameters.

    """
    # Use case 3: running robyn_inputs() with json_file
    if json_file is not None:
        json = robyn_read(json_file, step=1, **kwargs)
        if dt_input is None or dt_holidays is None:
            raise ValueError("Provide 'dt_input' and 'dt_holidays'")
        for i in range(len(json["InputCollect"])):
            assign(json["InputCollect"][i], json["InputCollect"][i])

    # Use case 1: running robyn_inputs() for the first time
    if InputCollect is None:
        ## dt_input = pd.as_tibble(dt_input), Manual, commenting out to see if tibble required
        if dt_holidays is not None:
            dt_holidays = dt_holidays  ## pd.as_tibble(dt_holidays) Manual, commenting out to see if tibble required
            # mutate(ds = as.Date(.data$ds, origin = "1970-01-01"))
            dt_holidays["ds"] = pd.to_datetime(dt_holidays["ds"])  ## Manually added
        else:
            dt_holidays = None

        # Check for NA values
        check_nas(dt_input)
        check_nas(dt_holidays)

        # Check vars names (duplicates and valid)
        check_varnames(
            dt_input,
            dt_holidays,
            dep_var,
            date_var,
            context_vars,
            paid_media_spends,
            organic_vars,
        )

        # Check date input (and set dayInterval and intervalType)
        date_input = check_datevar(dt_input, date_var)
        print(date_input)
        dt_input = date_input["dt_input"]  # sorted date by ascending
        date_var = date_input["date_var"]  # when date_var = "auto"
        dayInterval = date_input["dayInterval"]
        intervalType = date_input["intervalType"]

        # Check dependent var
        check_depvar(dt_input, dep_var, dep_var_type)

        # Check prophet
        if dt_holidays is None or prophet_vars is None:
            dt_holidays = prophet_vars = prophet_country = prophet_signs = None

        prophet_signs = check_prophet(
            dt_holidays, prophet_country, prophet_vars, prophet_signs, dayInterval
        )

        # Check baseline variables (and maybe transform context_signs)
        ## R lists has names, therefore, there is no need to get the "context_signs" field from context, can directly use the context_signs.
        ## context = check_context(dt_input, context_vars, context_signs)
        ## context_signs = context["context_signs"]
        context_signs = check_context(dt_input, context_vars, context_signs)

        # Check paid media variables (set mediaVarCount and maybe transform paid_media_signs)
        if paid_media_vars is None:
            paid_media_vars = paid_media_spends

        paidmedia = check_paidmedia(
            dt_input, paid_media_vars, paid_media_signs, paid_media_spends
        )
        paid_media_signs = paidmedia["paid_media_signs"]
        mediaVarCount = paidmedia["mediaVarCount"]
        ## exposure_vars = paid_media_vars - paid_media_spends, manually changed diff of lists didn't work.
        exposure_vars = np.setdiff1d(
            paid_media_vars, paid_media_spends
        )  ## Manually added

        # Check organic media variables (and maybe transform organic_signs)
        organic = check_organicvars(dt_input, organic_vars, organic_signs)
        organic_signs = organic["organic_signs"]

        # Check factor_vars
        factor_vars = check_factorvars(dt_input, factor_vars, context_vars, organic_vars)

        # Check all vars
        ## Manually changed to +
        ## all_media = [paid_media_spends, organic_vars]
        all_media = paid_media_spends + organic_vars
        ## Manually added following line to return prophet vars to lowercase
        prophet_vars = [(v.lower()) for v in prophet_vars]  ## Manually added
        ## all_ind_vars = [tolower(prophet_vars), context_vars, all_media]
        ## all_ind_vars = [prophet_vars, context_vars, all_media], this creates multip dimensional data, need list
        all_ind_vars = prophet_vars + context_vars + all_media
        check_allvars(all_ind_vars)

        # Check data dimension
        check_datadim(dt_input, all_ind_vars, rel=10)

        # Check window_start & window_end (and transform parameters/data)
        windows = check_windows(dt_input, date_var, all_media, window_start, window_end)
        dt_input = windows["dt_input"]
        window_start = windows["window_start"]
        rollingWindowStartWhich = windows["rollingWindowStartWhich"]
        refreshAddedStart = windows["refreshAddedStart"]
        window_end = windows["window_end"]
        rollingWindowEndWhich = windows["rollingWindowEndWhich"]
        rollingWindowLength = windows["rollingWindowLength"]

        # Check adstock
        adstock = check_adstock(adstock)

        # Check calibration and iters/trials
        calibration_input = check_calibration(
            dt_input,
            date_var,
            calibration_input,
            dayInterval,
            dep_var,
            window_start,
            window_end,
            paid_media_spends,
            organic_vars,
        )

        # Not used variables
        ## Manual corrections
        ## unused_vars = [var for var in colnames(dt_input) if var not in [dep_var, date_var, context_vars, paid_media_vars, paid_media_spends, organic_vars]]
        all_vars = [dep_var, date_var]
        all_vars += context_vars + paid_media_vars + paid_media_spends + organic_vars
        unused_vars = list()
        for var in dt_input.columns.values:
            if var not in all_vars:
                unused_vars.append(var)

        # Check for no-variance columns on raw data (after removing not-used)
        ## Manual correction needed, there is no use of select in python, or -all_of
        ## Meaning select the df except the unused_vars
        ## check_novar(select(dt_input, -all_of(unused_vars)))
        check_novar_columns = dt_input.columns.difference(unused_vars)

        check_novar(dt_input[check_novar_columns])

        # Calculate total media spend used to model
        ## Manual: there is no select method for Dataframe, and reverse order selection is not
        ## paid_media_total = dt_input[rollingWindowEndWhich:rollingWindowLength].select(paid_media_vars).sum()
        paid_media_df = dt_input.loc[
            (rollingWindowLength - 1):(rollingWindowEndWhich - 1), paid_media_vars
        ]
        paid_media_total = sum(
            paid_media_df.select_dtypes(include="number").sum(axis=0)
        )

        # Collect input
        InputCollect = {
            "dt_input": dt_input,
            "dt_holidays": dt_holidays,
            "dt_mod": None,
            "dt_modRollWind": None,
            "xDecompAggPrev": None,
            "date_var": date_var,
            "dayInterval": dayInterval,  ## Manually commented out since it is not NULL in the origiinal code. None,
            "intervalType": intervalType,  ## Manually commented out since it is not NULL in the origiinal code. None,
            "dep_var": dep_var,
            "dep_var_type": dep_var_type,
            "prophet_vars": prophet_vars,  ## Manually commented out since already done above. .lower(),
            "prophet_signs": prophet_signs,
            "prophet_country": prophet_country,
            "context_vars": context_vars,
            "context_signs": context_signs,
            "paid_media_vars": paid_media_vars,
            "paid_media_signs": paid_media_signs,
            "paid_media_spends": paid_media_spends,
            "paid_media_total": paid_media_total,  ## Manually commented out , it was None, since code inferred piece by piece it wasn't there
            "mediaVarCount": mediaVarCount,  ## Manually commented out , it was None, since code inferred piece by piece it wasn't there,
            "exposure_vars": exposure_vars,  ## Manually commented out , it was None, since code inferred piece by piece it wasn't there,,
            "organic_vars": organic_vars,
            "organic_signs": organic_signs,
            "all_media": all_media,  ## Manually commented out , it was None, since code inferred piece by piece it wasn't there,,
            "all_ind_vars": all_ind_vars,  ## Manually commented out , it was None, since code inferred piece by piece it wasn't there,,
            "factor_vars": factor_vars,
            "unused_vars": unused_vars,  ## Manually commented out , it was None, since code inferred piece by piece it wasn't there,,
            "window_start": window_start,
            "rollingWindowStartWhich": rollingWindowStartWhich,
            "window_end": window_end,
            "rollingWindowEndWhich": rollingWindowEndWhich,
            "rollingWindowLength": rollingWindowLength,
            "totalObservations": dt_input.shape[
                0
            ],  ## Manually commented out totalObservations, it is the count of rows in dt_input
            "refreshAddedStart": refreshAddedStart,
            "adstock": adstock,
            "hyperparameters": hyperparameters,
            "calibration_input": calibration_input,
            "custom_params": dict(),  ## custom_params, it is an empty list in original code
        }

        # Check hyperparameters
        ## Manual Fix ...
        if hyperparameters is not None:
            # Running robyn_inputs() for the 1st time & 'hyperparameters' provided --> run robyn_engineering()
            hyperparameters = check_hyperparameters(
                hyperparameters, adstock, paid_media_spends, organic_vars, exposure_vars
            )
            InputCollect = robyn_engineering(InputCollect, ...)

    else:
        # Check for legacy (deprecated) inputs
        check_legacy_input(InputCollect)
        # Check calibration data
        ## Manually corrected access to dict
        calibration_input = check_calibration(
            dt_input=InputCollect['dt_input'],
            date_var=InputCollect['date_var'],
            calibration_input=calibration_input,
            dayInterval=InputCollect['dayInterval'],
            dep_var=InputCollect['dep_var'],
            window_start=InputCollect['window_start'],
            window_end=InputCollect['window_end'],
            paid_media_spends=InputCollect['paid_media_spends'],
            organic_vars=InputCollect['organic_vars'],
        )

        # Update calibration_input
        if not calibration_input is None:
            InputCollect['calibration_input'] = calibration_input

        # Update hyperparameters
        if not hyperparameters is None:
            InputCollect['hyperparameters'] = hyperparameters

        # Check for hyperparameters
        ## if not InputCollect['hyperparameters'] and not hyperparameters:
        if not 'hyperparameters' in InputCollect.keys() and hyperparameters is None:
            raise ValueError("Must provide hyperparameters in robyn_inputs()")
        else:
        # Conditional output 2.1
        ## if InputCollect['hyperparameters']:
            if 'hyperparameters' in InputCollect.keys():
                # Update & check hyperparameters
                ## if not InputCollect['hyperparameters']
                if InputCollect['hyperparameters'] is None:
                    InputCollect['hyperparameters'] = hyperparameters
                InputCollect['hyperparameters'] = check_hyperparameters(
                    InputCollect['hyperparameters'],
                    InputCollect['adstock'],
                    InputCollect['all_media'],
                )

            # Run robyn_engineering()
            ## InputCollect = robyn_engineering(InputCollect, *args, **kwargs)
            InputCollect = robyn_engineering(InputCollect)

        # Check for no-variance columns (after filtering modeling window)
        select_columns = list(set([col for col in InputCollect['dt_mod'].columns.values if col not in InputCollect['unused_vars']]))
        temp_df = InputCollect['dt_mod'][select_columns]
        dt_mod_model_window = temp_df.loc[(temp_df['ds'] >= InputCollect['window_start']) & (temp_df['ds'] <= InputCollect['window_end'])]
        ##dt_mod_model_window = InputCollect['dt_mod'].select(
        ##    -InputCollect.unused_vars
        ##).filter(ds >= InputCollect.window_start, ds <= InputCollect.window_end)

        check_novar(dt_mod_model_window, InputCollect)

    # Handle JSON file input
    if json_file is not None:
        pending = [x for x in json_file["InputCollect"] if x not in InputCollect]
        InputCollect.extend(pending)

    # Save versions,
    ## Manual: Too R type of operation to save data in R studio.
    ## Manual: Versions etc.
    ## Manual: ver = pd.Series(utils.packageVersion('Robyn')).astype(str)
    ## Manual: rver = utils.sessionInfo()['R.version']
    ## Manual origin = 'dev' if utils.packageDescription('Robyn')['Repository'] is None else 'stable'
    ## InputCollect['version'] = f'Robyn ({origin}) v{ver} [R-{rver.major}.{rver.minor}]'

    # Set class
    ## Manual: Commenting out since return data should present robyn_inputs to InputCollect
    ## InputCollect['class'] = ['robyn_inputs', InputCollect['class']]
    ## return InputCollect
    return {"robyn_inputs": InputCollect}


def print_robyn_inputs(
    x,
    *,
    mod_vars=None,
    range=None,
    windows=None,
    custom_params=None,
    prophet=None,
    unused=None,
    hyps=None,
):
    """
    Print Robyn inputs.

    Args:
        x: The input data.
        mod_vars: The modified variables.
        range: The date range.
        windows: The model window.
        custom_params: The custom parameters.
        prophet: The Prophet configuration.
        unused: The unused variables.
        hyps: The hyper-parameter ranges.

    Returns:
        None
    """
    # Set mod_vars
    if mod_vars is None:
        mod_vars = ", ".join(setdiff(x.dt_mod.columns, ["ds", "dep_var"]))

    # Set range
    if range is None:
        range = pd.Timestamp(x.dt_input.iloc[0, 0]).strftime("%Y-%m-%d:%H:%M:%S")

    # Set windows
    if windows is None:
        windows = f"{x.window_start}:{x.window_end}"

    # Set custom_params
    if custom_params is None:
        custom_params = ""
    else:
        custom_params = ", ".join(x["custom_params"])

    # Set prophet
    if prophet is None:
        prophet = "Deactivated"
    else:
        prophet = f"{prophet} on {x.prophet_country}"

    # Set unused
    if unused is None:
        unused = "None"
    else:
        unused = ", ".join(x.unused_vars)

    # Set hyps
    if hyps is None:
        hyps = "Hyper-parameters: Not set yet"
    else:
        hyps = "Hyper-parameters ranges:\n" + "\n".join(x.hyperparameters)

    # Print Robyn inputs
    print(
        f"Total Observations: {x.nrow(x.dt_input)} ({x.intervalType}s)\n"
        f"Input Table Columns ({x.ncol(x.dt_input)}):\n"
        f"  Date: {x.date_var}\n"
        f"  Dependent: {x.dep_var} [{x.dep_var_type}]\n"
        f"  Paid Media: {', '.join(x.paid_media_vars)}\n"
        f"  Paid Media Spend: {', '.join(x.paid_media_spends)}\n"
        f"  Context: {', '.join(x.context_vars)}\n"
        f"  Organic: {', '.join(x.organic_vars)}\n"
        f"  Prophet (Auto-generated): {prophet}\n"
        f"  Unused variables: {unused}\n"
        f"  Date Range: {range}\n"
        f"  Model Window: {windows} ({x.rollingWindowEndWhich - x.rollingWindowStartWhich + 1} {x.intervalType}s)\n"
        f"  With Calibration: {x.calibration_input is not None}\n"
        f"  Custom parameters: {custom_params}\n"
        f"  Adstock: {x.adstock}\n"
        f"{hyps}"
    )


def robyn_engineering(x, quiet=False):
    """
    Performs feature engineering for the Robyn model.

    Args:
        x (dict): The input dictionary containing various data frames and variables.
        quiet (bool, optional): If True, suppresses the printing of recommendations and warnings. Defaults to False.

    Returns:
        tuple: A tuple containing the following:
            - mod_nls_collect (pd.DataFrame): A data frame containing the collected NLS models.
            - plot_nls_collect (pd.DataFrame): A data frame containing the collected NLS plots.
            - yhat_collect (pd.DataFrame): A data frame containing the collected yhat values.
    """
    print(">> Running Robyn feature engineering...")
    # InputCollect
    input_collect = x

    # check_InputCollect
    check_input_collect(input_collect)

    # dt_input
    ## dt_input = select(input_collect, -any_of(input_collect.unused_vars))
    used_columns = [var for var in input_collect['dt_input'].columns if var not in input_collect['unused_vars']]
    dt_input = input_collect['dt_input'][used_columns]

    # paid_media_vars
    paid_media_vars = input_collect['paid_media_vars']

    # paid_media_spends
    ##paid_media_spends = input_collect.paid_media_spends
    paid_media_spends = input_collect['paid_media_spends']

    # factor_vars
    ##factor_vars = input_collect.factor_vars
    factor_vars = input_collect['factor_vars']

    # rollingWindowStartWhich
    ##rolling_window_start_which = input_collect.rollingWindowStartWhich
    rolling_window_start_which = input_collect['rollingWindowStartWhich']

    # rollingWindowEndWhich
    ##rolling_window_end_which = input_collect.rollingWindowEndWhich
    rolling_window_end_which = input_collect['rollingWindowEndWhich']

    # dt_inputRollWind
    ##dt_input_roll_wind = dt_input[rolling_window_start_which:rolling_window_end_which,]
    dt_input_roll_wind = dt_input.loc[(rolling_window_start_which-1):(rolling_window_end_which-1),]

    # dt_transform
    dt_transform = dt_input
    ## colnames(dt_transform)[colnames(dt_transform) == input_collect.date_var] = "ds"
    ## colnames(dt_transform)[colnames(dt_transform) == input_collect.dep_var] = "dep_var"
    dt_transform = dt_transform.rename(columns={input_collect['date_var']:'ds', input_collect['dep_var']:'dep_var'})
    dt_transform = dt_transform.sort_values(by=['ds'])  ## manually changed
    # arrange(dt_transform, .data.ds) ## Manual what is this? arrange method used for sorting the data frame by date in R

    # dt_transformRollWind
    dt_transform_roll_wind = dt_transform.iloc[(rolling_window_start_which-1):(rolling_window_end_which-1),]

    # exposure_selector
    ## exposure_selector = paid_media_spends != paid_media_vars
    ## names(exposure_selector) = paid_media_vars ## Manual: names sets the column names of a data frame
    exposure_selector = list()
    for i, val in enumerate(paid_media_spends):
        ## exposure_selector[paid_media_vars[i]] = (val != paid_media_vars[i])
        exposure_selector.append(val != paid_media_vars[i])

    # modNLSCollect, plotNLSCollect, yhatCollect
    ## Manually added if case
    if any(exposure_selector):
        mod_nls_collect = []
        plot_nls_collect = []
        yhat_collect = []

        # mediaCostFactor
        # media_cost_factor = col_sums(dt_input_roll_wind[paid_media_spends], na.rm=True) / col_sums(dt_input_roll_wind[paid_media_vars], na.rm=True)
        ## media_cost_factor = col_sums(dt_input_roll_wind[paid_media_spends], True) / col_sums(dt_input_roll_wind[paid_media_vars], True)
        media_cost_factor = list()
        for i in range(len(paid_media_spends)):
            media_cost_factor.append(np.divide(np.sum(dt_input_roll_wind[paid_media_spends[i]]), np.sum(dt_input_roll_wind[paid_media_vars[i]])))

        # for loop
        ## for i in range(input_collect.mediaVarCount):
        for i in range(input_collect['mediaVarCount']):
            if exposure_selector[i]:
                # Run models (NLS and/or LM)
                ## dt_spend_mod_input = subset(dt_input_roll_wind, select=c(paid_media_spends[i], paid_media_vars[i]))
                dt_spend_mod_input = dt_input_roll_wind[[paid_media_spends[i], paid_media_vars[i]]]
                results = fit_spend_exposure(dt_spend_mod_input, media_cost_factor[i], paid_media_vars[i])

                # Compare NLS & LM, takes LM if NLS fits worse
                ##mod = results.res
                mod = results['res']
                ## Manual:
                exposure_selector[i] = False if mod['rsq_nls'] is None else mod['rsq_nls'][0] > mod['rsq_lm'][0]

                # Data to create plot
                dt_plot_nls = pd.DataFrame(
                    {
                        'channel': paid_media_vars[i],
                        'yhat_nls': results['yhatNLS'] if exposure_selector[i] else results['yhatLM'],  ## Manual : Wrong translation, manual correction
                        'yhat_lm': results['yhatLM'],
                        'y': results['data']['exposure'],
                        'x': results['data']['spend'],
                    }
                )

                caption = f"nls: AIC = {mod['aic_nls'].values[0]} | R2 = {mod['rsq_nls'].values[0]}\nlm: AIC = {mod['aic_lm'].values[0]} | R2 = {mod['rsq_lm'].values[0]}"

                ## dt_plot_nls = dt_plot_nls.pivot_longer(
                ##    cols=c("yhat_nls", "yhat_lm"), names_to="models", values_to="yhat")
                ## dt_plot_nls = pd.DataFrame({'yhat' = results['yhatNLS'] + results['yhatLM']})
                ## pivot_longer corresponding method is melt https://pandas.pydata.org/docs/reference/api/pandas.melt.html
                dt_plot_nls = pd.melt(dt_plot_nls, id_vars=['channel', 'y', 'x'], value_vars=['yhat_nls', 'yhat_lm'], var_name='models', value_name='yhat', ignore_index=False)
                ## dt_plot_nls["models"] = dt_plot_nls["models"].str.remove(
                ##    tolower(dt_plot_nls["models"]), "yhat"
                ##)
                ## + theme_lares(background="white", legend="top")
                ## ggplot in Python https://realpython.com/ggplot-python/
                models_plot = (
                    ggplot(dt_plot_nls)
                    + aes(x="x", y="y", color="models")
                    + geom_point()
                    + geom_line(aes(y="yhat", x="x", color="models"))
                    + labs(
                        title="Exposure-Spend Models Fit Comparison",
                        x=f"Spend [{paid_media_spends[i]}]",
                        y=f"Exposure [{paid_media_vars[i]}]",
                        caption=caption,
                        color="Model",
                        )
                        + theme_gray()
                        + scale_x_continuous()
                        + scale_y_continuous()
                )

                # Save results into modNLSCollect, plotNLSCollect, yhatCollect
                mod_nls_collect.append(mod)
                plot_nls_collect.append(models_plot)
                yhat_collect.append(dt_plot_nls)

        # bind_rows
        mod_nls_collect = pd.concat(mod_nls_collect)
        #plot_nls_collect = pd.concat(plot_nls_collect)
        yhat_collect = pd.concat(yhat_collect)
        repeat_factor = len(yhat_collect) // len(dt_transform_roll_wind)
        yhat_collect['ds'] = dt_transform_roll_wind['ds'].repeat(repeat_factor).reset_index(drop=True)

    else: ## Manually added else case wasn't translated, possibly due to large function.
        mod_nls_collect = None
        plot_nls_collect = None
        yhat_collect = None

    # Give recommendations and show warnings
    ## if not quiet:
    if mod_nls_collect is not None and not quiet:
        threshold = 0.8
        ## Manually added these
        these = None
        final_print = None ## False
        ## metrics = ["R2 (nls)", "R2 (lm)"]
        ## names = ["rsq_nls", "rsq_lm"]
        metrics = ["rsq_nls", "rsq_lm"]
        for m in range(len(metrics)):
            metric_name = metrics[m]
            temp = np.where(mod_nls_collect[metric_name] < threshold)[0]
            if len(temp) > 0:
                final_print = True
                ## these = x.iloc[temp, 0]
                these = x['channel'][temp]

        ## Manually corrected
        if final_print == True:
            print(f"NOTE: potential improvement on splitting channels for better exposure fitting.")
            print(f"Threshold (Minimum R2) = {threshold}")
            print(f"Check: InputCollect$modNLS$plots outputs")
            print(f"Weak relationship for: {v2t(these)} and their spend")

    # Clean & aggregate data
    #factor_vars = ["var1", "var2", "var3"]  # Replace with actual factor variables
    #if len(factor_vars) > 0:
    #    x = pd.get_dummies(x, drop_first=True, columns=factor_vars)
    x_df = pd.DataFrame(x['dt_input'])
    for col in factor_vars:
        if col in x_df.columns:
            x_df[col] = x_df[col].astype('category')


    # Initialize empty lists to store custom parameters and prophet arguments
    custom_params = dict()
    prophet_args = list()

    # Extract prophet variables and custom parameters from InputCollect
    prophet_vars = input_collect["prophet_vars"]
    custom_params = input_collect.get("custom_params", dict())

    # Remove empty strings and ellipsis from custom parameters
    ## custom_params = [param for name, param in custom_params.items() if param != "" and param != "..."]
    temp = dict()
    for param, val in custom_params.items():
        if not (param != "" and param != "..."):
            temp[param] = val

    custom_params = temp

    # Compute prophet arguments
    robyn_run_args = {
        "InputCollect", "dt_hyper_fixed", "json_file", "ts_validation",
        "add_penalty_factor", "refresh", "seed", "quiet", "cores", "trials",
        "iterations", "rssd_zero_penalty", "objective_weights",
        "nevergrad_algo", "intercept", "intercept_sign", "lambda_control", "outputs"
    }
    robyn_outputs_args = {
        "input_collect", "output_models", "pareto_fronts",
        "calibration_constraint", "plot_folder", "plot_folder_sub",
        "plot_pareto", "csv_out", "clusters", "select_model",
        "ui", "export", "all_sol_json", "quiet", "refresh"
    }
    robyn_inputs_args = {
        "dt_input", "dep_var", "dep_var_type", "date_var",
        "paid_media_spends", "paid_media_vars", "paid_media_signs",
        "organic_vars", "organic_signs", "context_vars", "context_signs",
        "factor_vars", "dt_holidays", "prophet_vars", "prophet_signs",
        "prophet_country", "adstock", "hyperparameters", "window_start",
        "window_end", "calibration_input", "json_file", "InputCollect"
    }
    robyn_refresh_args = {
        "json_file", "robyn_object", "dt_input", "dt_holidays",
        "refresh_steps", "refresh_mode", "refresh_iters", "refresh_trials",
        "plot_folder", "plot_pareto", "version_prompt", "export",
        "calibration_input", "objective_weights"
    }

    combined_args = (
        robyn_run_args |
        robyn_outputs_args |
        robyn_inputs_args |
        robyn_refresh_args
    )

    exclude_set = {"", "..."}

    robyn_args = combined_args - exclude_set

    # Compute custom prophet arguments
    prophet_custom_args = set(custom_params.keys()) - set(robyn_args)

    # Print message with custom prophet parameters
    if len(prophet_custom_args) > 0:
        print(f"Using custom prophet parameters: {', '.join(prophet_custom_args)}")

    # Decompose data using prophet
    dt_transform = prophet_decomp(
        dt_transform=dt_transform,
        dt_holidays=input_collect["dt_holidays"],
        prophet_country=input_collect["prophet_country"],
        prophet_vars=prophet_vars,
        prophet_signs=input_collect["prophet_signs"],
        factor_vars=input_collect["factor_vars"],
        context_vars=input_collect["context_vars"],
        organic_vars=input_collect["organic_vars"],
        paid_media_spends=input_collect["paid_media_spends"],
        intervalType=input_collect["intervalType"],
        dayInterval=input_collect["dayInterval"],
        custom_params=custom_params,
    )

    # Finalize enriched input
    ## dt_transform = dt_transform.subset(select=["ds", "dep_var", input_collect["all_ind_vars"]])
    subset = input_collect["all_ind_vars"]
    subset.append("ds")
    subset.append("dep_var")
    subset = list(set(subset))
    dt_transform = dt_transform[subset]

    input_collect["dt_mod"] = dt_transform
    input_collect["dt_modRollWind"] = dt_transform.iloc[(rolling_window_start_which-1):(rolling_window_end_which),]
    input_collect["dt_inputRollWind"] = dt_input_roll_wind
    input_collect["modNLS"] = {  ## Manual: added dict.
        "results": mod_nls_collect,
        "yhat": yhat_collect,
        "plots": plot_nls_collect
    }

    return input_collect


def prophet_decomp(
    dt_transform,
    dt_holidays,
    prophet_country,
    prophet_vars,
    prophet_signs,
    factor_vars,
    context_vars,
    organic_vars,
    paid_media_spends,
    intervalType,
    dayInterval,
    custom_params = dict(),
):
    """
    Decomposes a time series using the Prophet algorithm.

    Args:
        dt_transform (pandas.DataFrame): The input time series data.
        dt_holidays (pandas.DataFrame): The holiday data.
        prophet_country (str): The country for which the holidays are defined.
        prophet_vars (list): The variables to include in the Prophet model.
        prophet_signs (list): The signs of the variables in the Prophet model.
        factor_vars (list): The factor variables to include in the model.
        context_vars (list): The context variables to include in the model.
        organic_vars (list): The organic variables to include in the model.
        paid_media_spends (list): The paid media spends variables to include in the model.
        intervalType (str): The type of interval for the time series data.
        dayInterval (int): The interval between each data point.
        custom_params (dict, optional): Custom parameters for the Prophet model. Defaults to an empty dictionary.

    Returns:
        pandas.DataFrame: The transformed dataframe with decomposed time series.
    """
    # Check prophet
    check_prophet(
        dt_holidays, prophet_country, prophet_vars, prophet_signs, dayInterval
    )

    # Recurrence
    ## recurrence = dt_transform.select(["y" = "dep_var"]).rename(columns={"y": "dep_var"}) ## how to do select ???
    recurrence = dt_transform[["ds", "dep_var"]]
    recurrence.rename(columns={"dep_var": "y"}, inplace=True)

    # Holidays
    holidays = set_holidays(dt_transform, dt_holidays, intervalType)

    # Use trend, holiday, season, monthly, weekday, and weekly seasonality
    use_trend = True if "trend" in prophet_vars else False
    use_holiday = True if "holiday" in prophet_vars else False
    use_season = True if "season" in prophet_vars or "yearly.seasonality" in prophet_vars else False
    use_monthly = True if "monthly" in prophet_vars else False
    ## use_weekday = "weekday" in prophet_vars or "weekly.seasonality" in prophet_vars
    use_weekday = True if "weekday" in prophet_vars or "weekly.seasonality" in prophet_vars else False

    # Bind columns
    ## dt_regressors = pd.concat([recurrence,
    ##        pd.select(
    ##            dt_transform, all_of(c(paid_media_spends, context_vars, organic_vars))
    ##        ),
    ##    ],
    ##    axis=1,
    ##)
    dt_regressors = pd.concat([recurrence, dt_transform[paid_media_spends + context_vars + organic_vars]], axis=1)

    # Mutate date column
    dt_regressors["ds"] = pd.to_datetime(dt_regressors["ds"])

    # Prophet parameters
    prophet_params = {
        "holidays": holidays[holidays["country"] == prophet_country],
        "yearly_seasonality": custom_params["yearly_seasonality"] if "yearly_seasonality" in custom_params.keys() else use_season,
        "weekly_seasonality": custom_params["weekly_seasonality"] if "weekly_seasonality" in custom_params.keys() else use_weekday,
        "daily_seasonality": False, ## No hourly model allowed
    }

    custom_params["yearly_seasonality"] = None
    custom_params["weekly_seasonality"] = None
    # Append custom parameters
    ## prophet_params.update(custom_params)

    # Create prophet model
    ## https://facebook.github.io/prophet/docs/quick_start.html#python-api
    modelRecurrence = Prophet(**prophet_params)

    # Add seasonality
    if use_monthly:
        modelRecurrence.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    ## Manually added this section
    if factor_vars is not None and len(factor_vars) > 0:
        dt_ohse_temp = dt_regressors[factor_vars]
        dt_ohse = pd.get_dummies(dt_ohse_temp, columns=factor_vars, drop_first=False, dtype=int)

        temp_names = [col for col in dt_ohse.columns if not col.endswith('_na')]
        dt_ohse = dt_ohse[temp_names]
        ohe_names = list(dt_ohse.columns)

        for addreg in ohe_names:
            modelRecurrence.add_regressor(addreg)

        temp_names = [col for col in dt_regressors.columns if col not in factor_vars]
        dt_ohe = pd.concat([dt_regressors[temp_names], dt_ohse], axis=1)

        mod_ohe = modelRecurrence.fit(dt_ohe)
        dt_forecastRegressor = mod_ohe.predict(dt_ohe)
        select_columns = [column for column in dt_forecastRegressor.columns.values if not "_lower" in column or not "_upper" in column]
        forecastRecurrence = dt_forecastRegressor[select_columns]

        def custom_scale(array):
            scaler = StandardScaler()
            non_zero_mask = array != 0
            #scaled_non_zeros = scaler.fit_transform(array[non_zero_mask].reshape(-1, 1)).flatten()
            scaled_non_zeros = scaler.fit_transform(array[non_zero_mask].to_numpy().reshape(-1, 1)).flatten()

            scaled_array = np.copy(array)
            scaled_array[non_zero_mask] = scaled_non_zeros
            return scaled_array

        for aggreg in factor_vars:
            ohRegNames = [var for var in forecastRecurrence.columns.values if var.startswith(aggreg)]
            get_reg = forecastRecurrence[ohRegNames].sum(axis=1)
            dt_transform[aggreg] = custom_scale(get_reg)
    else:
        if dayInterval == 1:
            warnings.warn("Currently, there's a known issue with prophet that may crash this use case.\n Read more here: https://github.com/facebookexperimental/Robyn/issues/472")

        # Fit model
        mod = modelRecurrence.fit(dt_regressors)

        # Forecast
        forecastRecurrence = mod.predict(dt_regressors)


    # Add trend, season, monthly, and weekday features
    if use_trend:
        dt_transform["trend"] = forecastRecurrence['trend']
    if use_season:
        dt_transform["season"] = forecastRecurrence['yearly']
    if use_monthly:
        dt_transform["monthly"] = forecastRecurrence["monthly"]
    if use_weekday:
        dt_transform["weekday"] = forecastRecurrence["weekly"]
    if use_holiday:
        dt_transform["holiday"] = forecastRecurrence["holidays"]

    # Return transformed dataframe
    return dt_transform


def fit_spend_exposure(dt_spendModInput, mediaCostFactor, paid_media_var):
    """
    Fits the spend-exposure model using two different approaches: Michaelis-Menten model and linear model comparison.

    Parameters:
    - dt_spendModInput (DataFrame): Input data with two columns: 'spend' and 'exposure'.
    - mediaCostFactor (float): Factor to adjust the media cost.
    - paid_media_var (str): Name of the paid media variable.

    Returns:
    - output (dict): Dictionary containing the model results and other relevant information.
    """
    # Check if the input data has the correct shape
    if dt_spendModInput.shape[1] != 2:
        raise ValueError("Pass only 2 columns")

    # Set the column names
    dt_spendModInput = dt_spendModInput.rename(columns = {dt_spendModInput.columns.values[0] : "spend", dt_spendModInput.columns.values[1]: "exposure"})

    # Model 1: Michaelis-Menten model Vmax * spend/(Km + spend)
    try:
        # Initialize the model parameters
        nlsStartVal = { ## Manually converted to dict
            "Vmax" : max(dt_spendModInput["exposure"]),
            "Km" : max(dt_spendModInput["exposure"]) / 2,
        }

        # Fit the model using non-linear least squares
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        ## https://www.rdocumentation.org/packages/minpack.lm/versions/1.2-4/topics/nlsLM
        ## https://lmfit.github.io/lmfit-py/model.html
        ## https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult

        ## nlsLM in R corresponds to lmfit in python
        def michaelis_menten(params, spend, exposure):
            vmax = params["vmax"]
            km = params["km"]
            model = vmax * spend / (km + spend)
            return exposure - model

        ## modNLS = curve_fit(michaelis_menten, dt_spendModInput["spend"], dt_spendModInput["exposure"], method='lm')
        ## popt, pconv, infodict, mesg, ier = modNLS

        start_params = create_params(vmax = nlsStartVal["Vmax"], km = nlsStartVal["Km"])
        modNLS = minimize(michaelis_menten, start_params, method='leastsq', args=(dt_spendModInput["spend"], dt_spendModInput["exposure"]) )
        ## modNLSparams = modNLS.make_params(vmax = nlsStartVal["Vmax"], km = nlsStartVal["Km"])

        # Get the predicted values
        ## yhatNLS = modNLS.predict()
        def predict_nls(spend, vmax, km):
            return vmax * spend / (km + spend)


        yhatNLS = predict_nls(dt_spendModInput["spend"], modNLS.params['vmax'].value, modNLS.params['km'].value)

        ## modNLSSum = modNLS.fit(dt_spendModInput["exposure"], params=modNLSparams, spend=dt_spendModInput["spend"])
        ## modNLS = nlsLM(
        ##    exposure=nlsStartVal["Vmax"]
        ##    * dt_spendModInput["spend"]
        ##    / (nlsStartVal["Km"] + dt_spendModInput["spend"]),
        ##    data=dt_spendModInput,
        ##    start=nlsStartVal,


        # Get the model summary
        ## modNLSSum = summary(modNLS)
        modNLSSum = modNLS

        # Calculate the R-squared value
        rsq_nls = r2_score(dt_spendModInput["exposure"], yhatNLS)

        # Check if the model is a good fit
        if rsq_nls < 0.7:
            print(
                f"Spend-exposure fitting for {paid_media_var} has rsq = {round(rsq_nls, 4)}. To increase the fit, try splitting the variable. Otherwise consider using spend instead."
            )
    except Exception as e:
        print(f"Error fitting Michaelis-Menten model: {e}")
        modNLS = None
        yhatNLS = None
        modNLSSum = None
        rsq_nls = None

    # Model 2: Build linear model comparison model
    ## Manually add the callback model
    ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html#scipy.stats.linregress
    ## https://lmfit.github.io/lmfit-py/builtin_models.html

    ## modLM = LinearRegression().fit(exposure=dt_spendModInput["spend"] - 1, data=dt_spendModInput)
    ## modLM = linregress(linear_model(dt_spendModInput["spend"]), dt_spendModInput["exposure"])
    ## def predict_linear_model(spend, lm_model):
        ## return lm_model.intercept + lm_model.slope * spend
    ## yhatLM = modLM.predict()

    modLM = LinearModel()
    lm_params = modLM.make_params(intercept=-1, slope=1)
    modLMSum = modLM.fit(dt_spendModInput["exposure"], lm_params, x=dt_spendModInput["spend"])

    ## modLMSum = summary(modLM)
    yhatLM = modLM.eval(lm_params, x=dt_spendModInput["spend"])
    ##rsq_lm = modLMSum.r2_score
    ##rsq_lm = modLMSum.rvalue ** 2
    rsq_lm = modLMSum.rsquared

    ## Manually added
    if rsq_lm is None or np.isnan(rsq_lm):
        raise ValueError(f"Check if {paid_media_var} contains only zeros")

    # Calculate the AIC and BIC values
    ## aic_nls = AIC(modNLS) if modNLS else np.nan
    ## aic_lm = AIC(modLM)
    ## bic_nls = BIC(modNLS) if modNLS else np.nan
    ## bic_lm = BIC(modLM)

    # Create the output dictionary
    output = {
        "res": pd.DataFrame(
            {
                ## "channel": [paid_media_var], corrected outputs
                "channel": [paid_media_var],
                "Vmax": [modNLS.params['vmax'].value if modNLS else np.nan],
                "Km": [modNLS.params['km'].value if modNLS else np.nan],
                "aic_nls": [modNLS.aic if modNLS is not None else None],
                "aic_lm": [modLMSum.aic],
                "bic_nls": [modNLS.bic if modNLS is not None else None],
                "bic_lm": [modLMSum.bic],
                "rsq_nls": [rsq_nls if modNLS is not None else None],
                "rsq_lm": [rsq_lm],
                "coef_lm": [modLMSum.summary()['best_values']['slope']],
            }
        ),
        "yhatNLS": yhatNLS,
        "modNLS": modNLS,
        "yhatLM": yhatLM,
        "modLM": modLM,
        "data": dt_spendModInput,
        "type": ["mm" if modNLS else "lm"],
    }

    return output


def set_holidays(dt_transform, dt_holidays, intervalType):
    """
    Sets the holidays based on the given interval type.

    Args:
        dt_transform (DataFrame): The transformed date DataFrame.
        dt_holidays (DataFrame): The holidays DataFrame.
        intervalType (str): The interval type. Can be one of: "day", "week", "month".

    Returns:
        DataFrame: The holidays DataFrame with aggregated holiday information.

    Raises:
        ValueError: If the intervalType is not one of the valid options.
        ValueError: If the week start is not Monday or Sunday for the "week" intervalType.
        ValueError: If the monthly data does not have the first day of the month as the datestamp.

    """
    opts = ["day", "week", "month"]
    if intervalType not in opts:
        raise ValueError(
            "Pass a valid 'intervalType'. Any of: {}, {}".format(opts, intervalType)
        )

    if intervalType == "day":
        holidays = dt_holidays
    elif intervalType == "week":
        ## week_start = dt_transform.ds.dt.weekday(1)
        week_start = dt_transform['ds'].dt.dayofweek[0] + 1
        if week_start not in [1, 7]:
            raise ValueError("Week start has to be Monday or Sunday")

        ## holidays = dt_holidays['ds'].dt.floor(
        ##    dt.Date(dt_transform.ds.dt.strftime("%Y-%m-%d"), origin="1970-01-01"),
        ##    unit="week",
        ##    week_start=week_start,
        ##)
        dt_holidays['ds'] = dt_holidays['ds'] - pd.to_timedelta(dt_holidays['ds'].dt.dayofweek, unit='d')

        ## holidays = holidays.dt.select(
        ##    holidays.ds, holidays.holiday, holidays.country, holidays.year
        ##)
        holidays = dt_holidays[['ds', 'holiday', 'country', 'year']]

        ##holidays = holidays.dt.groupby(
        ##    holidays.ds, holidays.country, holidays.year
        ##).agg({"holiday": lambda x: ", ".join(x), "n": "count"})

        holidays = holidays.groupby(['ds', 'country', 'year']).agg({'holiday':[' ,'.join, 'count']}).reset_index()
        temp = pd.DataFrame()
        temp['ds'] = holidays['ds']
        temp['country'] = holidays['country']
        temp['year'] = holidays['year']
        temp['holiday'] = holidays['holiday']['join']
        temp['n'] = holidays['holiday']['count']
        holidays = temp

    elif intervalType == "month":
        if not all(dt_transform['ds'].dt.is_month_start):
            raise ValueError(
                "Monthly data should have first day of month as datestampe, e.g.'2020-01-01'"
            )
        dt_holidays['ds'] = dt_holidays['ds'] - pd.to_timedelta(dt_holidays['ds'].dt.days_in_month, unit='d')
        holidays = holidays[['ds', 'holiday', 'country', 'year']]
        holidays = holidays.groupby(['ds', 'country', 'year']).agg({'holiday':[' ,'.join, 'count']}).reset_index()
        temp = df.DataFrame()
        temp['ds'] = holidays['ds']
        temp['country'] = holidays['country']
        temp['year'] = holidays['year']
        temp['holiday'] = holidays['holiday']['join']
        temp['n'] = holidays['holiday']['count']
        holidays = temp

    return holidays
