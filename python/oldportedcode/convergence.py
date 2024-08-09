# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, scale_colour_gradient, labs, theme, guides

def robyn_converge(OutputModels, n_cuts=20, sd_qtref=3, med_lowb=2, nrmse_win=(0, 0.998), **kwargs):
    assert n_cuts > min(sd_qtref, med_lowb) + 1, "n_cuts must be greater than min(sd_qtref, med_lowb) + 1"

    df_list = [trial['resultCollect']['resultHypParam'] for trial in OutputModels["trials"]]
    df = pd.concat(df_list, ignore_index=True)
    calibrated = df['mape'].sum() > 0

    # Calculate deciles
    df_melt = pd.melt(df, id_vars=['trial', 'ElapsedAccum'], value_vars=["nrmse", "decomp.rssd", "mape"], var_name="error_type", value_name="value")
    df_melt['error_type'] = df_melt['error_type'].str.upper()
    df_melt = df_melt[(df_melt['value'] > 0) & np.isfinite(df_melt['value'])]

    df_melt.sort_values(by=['trial', 'ElapsedAccum'], inplace=True)
    df_melt.reset_index(drop=True, inplace=True)
    df_melt['iter'] = df_melt.groupby(['error_type', 'trial']).cumcount() + 1
    max_iter = df_melt['iter'].max()
    cuts_labels = range(1, n_cuts + 1)
    df_melt['cuts'] = pd.cut(df_melt['iter'], bins=np.linspace(0, max_iter, n_cuts + 1), labels=cuts_labels, include_lowest=True, ordered=False)



    # Assuming 'ElapsedAccum' and 'trial' columns exist in df for sorting and grouping
    df_melt.sort_values(by=['trial', 'ElapsedAccum'], inplace=True)
    df_melt['iter'] = df_melt.groupby(['error_type', 'trial']).cumcount() + 1
    max_iter = df_melt['iter'].max()
    cuts_labels = range(1, n_cuts + 1)
    df_melt['cuts'] = pd.cut(df_melt['iter'], bins=np.linspace(0, max_iter, n_cuts+1), labels=cuts_labels, include_lowest=True, ordered=True)

    # print(df_melt)
    grouped = df_melt.groupby(['error_type', 'cuts'])
    errors = grouped.agg(
        n=('value', 'size'),
        median=('value', 'median'),
        std=('value', 'std')
    ).reset_index()
    errors['med_var_P'] = errors.groupby('error_type')['median'].transform(lambda x: abs(round(100 * (x - x.shift()) / x, 2)))

    errors['first_med'] = errors.groupby('error_type')['median'].transform('first').abs()
    errors['first_med_avg'] = errors.groupby('error_type').apply(lambda x: abs(x['median'].iloc[:sd_qtref].mean())).reset_index(level=0, drop=True)
    errors['last_med'] = errors.groupby('error_type')['median'].transform('last').abs()
    errors['first_sd'] = errors.groupby('error_type')['std'].transform('first')
    errors['first_sd_avg'] = errors.groupby('error_type')['std'].transform(lambda x: x.iloc[:sd_qtref].mean())

    errors['last_sd'] = errors.groupby('error_type')['std'].transform('last')
    errors['med_thres'] = abs(errors['first_med'] - med_lowb * errors['first_sd_avg'])
    errors['flag_med'] = errors['median'].abs() < errors['med_thres']
    errors['flag_sd'] = errors['std'] < errors['first_sd_avg']


    conv_msg = []
    unique_error_types = errors['error_type'].unique()

    for obj_fun in unique_error_types:
        temp_df = errors[errors['error_type'] == obj_fun].copy()
        temp_df['median'] = temp_df['median'].round(decimals=2)
        last_row = temp_df.iloc[-1]
        greater = ">"  # Equivalent to the R's intToUtf8(8814)

        # Constructing the message
        did_converge = "" if (last_row['flag_sd'] & last_row['flag_med']) else "NOT "
        sd = round(last_row['last_sd'], 2)
        symb_sd = "<=" if last_row['flag_sd'] else greater
        sd_thresh = round(last_row['first_sd_avg'], 2)
        quantile = 'n_cuts'  # Assuming n_cuts is defined somewhere in your context
        qtn_median = round(last_row['last_med'], 2)
        symb_med = "<=" if last_row['flag_med'] else greater
        med_thresh = round(last_row['med_thres'], 2)

        message = f"{last_row['error_type']} {did_converge}converged: sd@qt.{quantile} {sd} {symb_sd} {sd_thresh} & |med@qt.{quantile}| {qtn_median} {symb_med} {med_thresh}"
        conv_msg.append(message)

    for msg in conv_msg:
        print("-", msg)

    max_trial = df['trial'].max()
    trials_word = "trials" if max_trial > 1 else "trial"
    iterations_word = "each" if max_trial > 1 else ""
    nevergrad_algo = OutputModels['metadata']['nevergrad_algo']  # Assuming this is a dictionary and 'nevergrad_algo' is a key
    subtitle = f"{max_trial} {trials_word} with {df_melt['cuts'].max()} iterations {iterations_word} using {nevergrad_algo}"

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(df_melt, row='error_type', hue='error_type', aspect=15, height=0.5, palette='GnBu')
    g.map(sns.kdeplot, 'value', bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5).add_legend()
    g.map(sns.kdeplot, 'value', clip_on=False, color="w", lw=2, bw_adjust=0.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    for ax, title in zip(g.axes.flat, df_melt['error_type'].unique()):
        ax.set_title(title)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Objective convergence by iterations quantiles", fontsize=16)
    g.fig.subplots_adjust(top=.92)
    g.fig.suptitle(subtitle)

    moo_cloud_plot = (ggplot(df, aes(x='nrmse', y='decomp.rssd', colour='ElapsedAccum'))
                  + scale_colour_gradient(low="skyblue", high="#000080")
                  + labs(title="Multi-objective evolutionary performance" + (" with calibration" if calibrated else ""),
                         subtitle=subtitle,
                         x="NRMSE" + (" [Winsorized]" if max(df['nrmse']) == 1 else ""),
                         y="DECOMP.RSSD",
                         colour="Time [s]",
                         caption='\n'.join(conv_msg))  # Assuming conv_msg is a list of strings
                  + theme(figure_size=(10, 6)))

    if calibrated:
        moo_cloud_plot += (geom_point(aes(size='mape', alpha=1 - df['mape'])))
        moo_cloud_plot += guides(alpha=False, size=False)  # Correctly apply guides at the plot level
    else:
        moo_cloud_plot += geom_point()

    cvg_out = {
        'moo_distrb_plot': g,
        'moo_cloud_plot': moo_cloud_plot,
        'errors': errors,
        'conv_msg': conv_msg
    }

    cvg_out['sd_qtref'] = sd_qtref
    cvg_out['med_lowb'] = med_lowb

    return cvg_out


import matplotlib.pyplot as plt

# test coverage ...
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma


def gamma_mle(params, x):
    gamma_shape, gamma_scale = params
    return -np.sum(gamma.logpdf(x, shape=gamma_shape, scale=gamma_scale))


def f_geo(a, r, n):
    for i in range(2, n):
        a[i] = a[i - 1] * r
    return a


def nloptr(x0, eval_f, lb, x, opts):
    return minimize(eval_f, x0, method="SLSQP", bounds=lb, x=x, **opts)


def test_cvg():
    # Experiment with gamma distribution fitting

    # Initialize parameters
    gamma_shape = 5
    gamma_scale = 0.7

    # Generate sequence for fitting
    seq_nrmse = f_geo(5, 0.7, 100)

    # Create data frame with true values
    df_nrmse = pd.DataFrame({"x": range(1, 101), "y": seq_nrmse, "type": "true"})

    # Fit gamma distribution using NLOPT
    mod_gamma = nloptr(
        x0=[gamma_shape, gamma_scale],
        eval_f=gamma_mle,
        lb=[0, 0],
        x=seq_nrmse,
        opts={"algorithm": "SLSQP", "maxeval": 1e5},
    )

    # Extract fitted parameters
    gamma_params = mod_gamma.x

    # Generate predicted values
    seq_nrmse_gam = 1 / gamma.pdf(
        seq_nrmse, shape=gamma_params[0], scale=gamma_params[1]
    )
    seq_nrmse_gam = seq_nrmse_gam / (max(seq_nrmse_gam) - min(seq_nrmse_gam))
    seq_nrmse_gam = max(seq_nrmse) * seq_nrmse_gam

    # Create data frame with predicted values
    df_nrmse_gam = pd.DataFrame(
        {"x": range(1, 101), "y": seq_nrmse_gam, "type": "pred"}
    )

    # Combine data frames
    df_nrmse = pd.concat([df_nrmse, df_nrmse_gam], ignore_index=True)

    # Plot true and predicted values
    plt.plot(df_nrmse["x"], df_nrmse["y"], color="blue", label="True")
    plt.plot(df_nrmse["x"], df_nrmse_gam["y"], color="red", label="Predicted")
    plt.legend()
    plt.show()

    return df_nrmse
