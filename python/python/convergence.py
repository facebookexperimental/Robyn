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
    nevergrad_algo = OutputModels['nevergrad_algo']  # Assuming this is a string
    subtitle = f"{max_trial} {trials_word} with {dt_objfunc_cvg['cuts'].max()} iterations {iterations_word} using {nevergrad_algo}"

    moo_distrb_plot = (ggplot(dt_objfunc_cvg, aes(x='value', y='cuts', fill='-cuts'))
                   + geom_density_ridges(scale=2.5, color="white", quantile_lines=True, quantiles=2, alpha=0.7)
                   + facet_grid('. ~ error_type', scales="free")
                   + scale_fill_distiller(palette="GnBu")
                   + labs(title="Objective convergence by iterations quantiles",
                          x="Objective functions", y="Iterations [#]",
                          subtitle=subtitle)
                   + theme(figure_size=(10, 6)))

    if calibrated:
        moo_cloud_plot += (geom_point(aes(size='mape', alpha=1 - df['mape']), data=df)
                        + guides(alpha=False))
    else:
        moo_cloud_plot += geom_point()

    # Constructing the output dictionary
    cvg_out = {
        'moo_distrb_plot': moo_distrb_plot,
        'moo_cloud_plot': moo_cloud_plot,
        'errors': errors,  # Assuming errors is a DataFrame or similar object
        'conv_msg': conv_msg
    }

    # Adding attributes to the dictionary (or you can handle these separately)
    cvg_out['sd_qtref'] = sd_qtref
    cvg_out['med_lowb'] = med_lowb

    # Return the dictionary, note in Python there's no need for 'invisible'
    return cvg_out

# def robyn_converge(OutputModels, n_cuts=20, sd_qtref=3, med_lowb=2, nrmse_win=None):
#     # Stop if n_cuts is not valid
#     if n_cuts < sd_qtref + 1:
#         raise ValueError("n_cuts must be greater than or equal to sd_qtref + 1")

#     # Gather all trials
#     get_trials = np.where(
#         np.in1d(names(OutputModels), ["trial"] + np.arange(OutputModels.shape[1]))
#     )[0]
#     df = pd.concat([OutputModels[g] for g in get_trials], ignore_index=True)

#     # Calculate deciles
#     dt_objfunc_cvg = df.melt(
#         id_vars=["trial", "error_type"], value_vars=["nrmse", "decomp.rssd", "mape"]
#     )
#     dt_objfunc_cvg = dt_objfunc_cvg.dropna()
#     dt_objfunc_cvg["error_type"] = dt_objfunc_cvg["error_type"].astype(str)
#     dt_objfunc_cvg["cuts"] = pd.cut(
#         dt_objfunc_cvg["iter"],
#         bins=np.linspace(0, max(dt_objfunc_cvg["iter"]), n_cuts + 1),
#         labels=round(np.linspace(max(dt_objfunc_cvg["iter"]), 0, n_cuts), 2),
#     )
#     dt_objfunc_cvg = dt_objfunc_cvg.groupby(["error_type", "cuts"]).agg(
#         {"value": ["mean", "std"]}
#     )
#     dt_objfunc_cvg = dt_objfunc_cvg.drop(columns=["value"])
#     dt_objfunc_cvg.columns = ["error_type", "cuts", "mean", "std"]

#     # Calculate standard deviations and absolute medians on each cut
#     errors = dt_objfunc_cvg.groupby(["error_type", "cuts"]).agg(
#         {"mean": ["median"], "std": ["sd"]}
#     )
#     errors["med_var_P"] = np.abs(
#         round(
#             100 * (errors["median"] - errors["median"].shift(1)) / errors["median"], 2
#         )
#     )
#     errors["first_med"] = np.abs(errors["median"].shift(1))
#     errors["first_med_avg"] = np.mean(errors["median"][1:sd_qtref])
#     errors["last_med"] = np.abs(errors["median"].shift(-1))
#     errors["last_sd"] = np.abs(errors["std"].shift(-1))
#     errors["med_thres"] = np.abs(
#         errors["first_med"] - med_lowb * errors["first_sd_avg"]
#     )
#     errors["flag_med"] = np.abs(errors["median"]) < errors["med_thres"]
#     errors["flag_sd"] = errors["std"] < errors["first_sd_avg"]

#     # Create a convergence message
#     conv_msg = []
#     for obj_fun in np.unique(errors["error_type"]):
#         temp_df = errors[errors["error_type"] == obj_fun]
#         temp_df["median"] = np.sign(temp_df["median"], 2)
#         last_qt = temp_df.tail(1)
#         greater = ">"
#         temp = glue(
#             "{error_type} {did}converged: sd@qt.{quantile} {sd} {symb_sd} {sd_threh} &"
#         )
#         temp = temp.format(
#             error_type=last_qt["error_type"],
#             did=np.where(last_qt["flag_sd"] & last_qt["flag_med"], "", "NOT "),
#             sd=np.sign(last_qt["last_sd"], 2),
#             symb_sd=np.where(last_qt["flag_sd"], "<=", greater),
#             quantile=n_cuts,
#             qtn_median=np.sign(last_qt["last_med"], 2),
#             symb_med=np.where(last_qt["flag_med"], "<=", greater),
#             med_thresh=np.sign(last_qt["med_thres"], 2),
#         )
#         conv_msg.append(temp)

#     # Create a plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(
#         dt_objfunc_cvg.melt(id_vars=["error_type", "cuts"], value_vars=["mean", "std"]),
#         ax=ax,
#         cmap="coolwarm",
#     )
#     ax.set_title("Objective convergence by iterations quantiles")
#     ax.set_subtitle("")
#     ax.set_xlabel("Objective functions")
#     ax.set_ylabel("Iterations [#]")
#     ax.set_xlim([-1, max(dt_objfunc_cvg["cuts"])])
#     ax.set_ylim([-1, max(dt_objfunc_cvg["iter"])])
#     ax.set_xticks([i for i in range(n_cuts)])
#     ax.set_yticks([i for i in range(max(dt_objfunc_cvg["iter"]))])
#     ax.set_xticklabels(dt_objfunc_cvg["cuts"].values)
#     ax.set_yticklabels(dt_objfunc_cvg["iter"].values)
#     ax.set_title("Objective convergence by iterations quantiles")
#     ax.set_subtitle("")
#     ax.set_xlabel("Objective functions")
#     ax.set_ylabel("Iterations [#]")
#     ax.set_xlim([-1, max(dt_objfunc_cvg["cuts"])])
#     ax.set_ylim([-1, max(dt_objfunc_cvg["iter"])])
#     ax.set_xticks([i for i in range(n_cuts)])
#     ax.set_yticks([i for i in range(max(dt_objfunc_cvg["iter"]))])
#     ax.set_xticklabels(dt_objfunc_cvg["cuts"].values)
#     ax.set_yticklabels(dt_objfunc_cvg["iter"].values)
#     plt.show()

#     return conv_msg


# import matplotlib.pyplot as plt
# import numpy as np

# ### Rest of the inference:


# import pandas as pd
# from matplotlib.colors import ListedColormap
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler


# def robyn_converge(
#     OutputModels, n_cuts=20, sd_qtref=3, med_lowb=2, nrmse_win=None, **kwargs
# ):
#     # Convert the R code's 'df' to a Pandas DataFrame
#     df = pd.DataFrame(OutputModels)

#     # Calculate the Winsorized NRMSE
#     nrmse_win = np.array(nrmse_win)
#     nrmse = np.zeros(len(df))
#     for i in range(len(df)):
#         nrmse[i] = np.minimum(nrmse_win[0], np.maximum(nrmse_win[1], df.iloc[i, 0]))
#     df["nrmse"] = nrmse

#     # Create the main plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(
#         df.iloc[:, 1:],
#         cmap="coolwarm",
#         square=True,
#         xticks=range(len(df.columns)),
#         yticks=range(len(df.columns)),
#         annot=True,
#         ax=ax,
#     )
#     ax.set_title("Multi-objective evolutionary performance")
#     ax.set_xlabel("NRMSE")
#     ax.set_ylabel("DECOMP.RSSD")
#     ax.set_zlabel("Time [s]")
#     ax.set_size_label("MAPE")
#     ax.set_alpha_label(None)
#     ax.set_caption(conv_msg)

#     # Add a point cloud for the calibrated model
#     if calibrated:
#         ax.scatter(
#             df.iloc[:, 0],
#             df.iloc[:, 1],
#             c=df.iloc[:, 2],
#             alpha=1 - df.iloc[:, 2],
#             s=df.iloc[:, 3],
#             linewidths=1,
#             alpha=0.5,
#         )
#     else:
#         ax.scatter(
#             df.iloc[:, 0],
#             df.iloc[:, 1],
#             c=df.iloc[:, 2],
#             alpha=1,
#             s=df.iloc[:, 3],
#             linewidths=1,
#         )

#     # Add a color bar
#     cmap = ListedColormap(["skyblue", "navyblue"])
#     norm = Normalize(vmin=0, vmax=1)
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array(df.iloc[:, 2])
#     ax.colorbar(sm)

#     # Add a title and labels
#     ax.set_title("Multi-objective evolutionary performance")
#     ax.set_xlabel("NRMSE")
#     ax.set_ylabel("DECOMP.RSSD")
#     ax.set_zlabel("Time [s]")
#     ax.set_size_label("MAPE")
#     ax.set_alpha_label(None)
#     ax.set_caption(conv_msg)

#     # Create a list to store the plot and other outputs
#     cvg_out = []

#     # Add the plot to the list
#     cvg_out.append(fig)

#     # Add other outputs to the list
#     cvg_out.append(df)
#     cvg_out.append(errors)
#     cvg_out.append(conv_msg)

#     # Return the list
#     return cvg_out


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
