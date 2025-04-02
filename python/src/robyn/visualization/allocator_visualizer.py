import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Union
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from robyn.allocator.allocator import BudgetAllocator
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class AllocatorVisualizer(BaseVisualizer):
    """Generates plots for Robyn allocator results."""

    def __init__(
        self,
        budget_allocator: BudgetAllocator,
        metric: str = "ROAS",
        quiet: bool = False,
    ):
        super().__init__()
        logger.info("Initializing AllocatorPlotter")
        self.dt_optim_out = budget_allocator.dt_optim_out
        self.eval_list = budget_allocator.eval_dict
        self.metric = metric
        self.budget_allocator = budget_allocator
        self.logger = logging.getLogger(__name__)

    def _plot_response_spend_comparison(self) -> go.Figure:
        """Creates the response and spend comparison plot."""
        # Initial values
        init_total_spend = self.dt_optim_out["initSpendTotal"].iloc[0]
        init_total_response = self.dt_optim_out["initResponseTotal"].iloc[0]
        init_total_roi = init_total_response / init_total_spend
        init_total_cpa = init_total_spend / init_total_response
        # Bounded optimization values
        optm_total_spend_bounded = self.dt_optim_out["optmSpendTotal"].iloc[0]
        optm_total_response_bounded = self.dt_optim_out["optmResponseTotal"].iloc[0]
        optm_total_roi_bounded = optm_total_response_bounded / optm_total_spend_bounded
        optm_total_cpa_bounded = optm_total_spend_bounded / optm_total_response_bounded
        # Unbounded optimization values
        optm_total_spend_unbounded = self.dt_optim_out["optmSpendTotalUnbound"].iloc[0]
        optm_total_response_unbounded = self.dt_optim_out[
            "optmResponseTotalUnbound"
        ].iloc[0]
        optm_total_roi_unbounded = (
            optm_total_response_unbounded / optm_total_spend_unbounded
        )
        optm_total_cpa_unbounded = (
            optm_total_spend_unbounded / optm_total_response_unbounded
        )
        bound_mult = self.dt_optim_out["unconstr_mult"].iloc[0]

        # Check if optimization topped out
        optm_topped_bounded = optm_topped_unbounded = any_topped = False
        if self.eval_list.get("total_budget") is not None:
            total_budget = self.eval_list["total_budget"]
            optm_topped_bounded = round(optm_total_spend_bounded) < round(total_budget)
            optm_topped_unbounded = round(optm_total_spend_unbounded) < round(
                total_budget
            )
            any_topped = optm_topped_bounded or optm_topped_unbounded
            if optm_topped_bounded and not self.quiet:
                print(
                    "NOTE: Given the upper/lower constrains, the total budget can't be fully allocated (^)"
                )

        # Get levs1 from eval_list
        levs1 = self.eval_list.get(
            "levs1", ["Initial", "Bounded", f"Bounded x{bound_mult}"]
        )

        # If second and third levels are the same, add a space to third level
        if levs1[1] == levs1[2]:
            levs1[2] = f"{levs1[2]} "

        # Create levs2 based on scenario
        if self.budget_allocator.params.scenario == "max_response":
            levs2 = [
                "Initial",
                f"Bounded{'ˆ' if optm_topped_bounded else ''}",
                f"Bounded{'ˆ' if optm_topped_unbounded else ''} x{bound_mult}",
            ]
        else:  # target_efficiency
            levs2 = levs1

        # Create response metric dataframe
        self.resp_metric = pd.DataFrame(
            {
                "type": pd.Categorical(
                    levs1, categories=levs1
                ),  # Make it a factor with levels
                "type_lab": pd.Categorical(
                    levs2, categories=levs2
                ),  # Make it a factor with levels
                "total_spend": [
                    init_total_spend,
                    optm_total_spend_bounded,
                    optm_total_spend_unbounded,
                ],
                "total_response": [
                    init_total_response,
                    optm_total_response_bounded,
                    optm_total_response_unbounded,
                ],
                "total_response_lift": [
                    0,
                    self.dt_optim_out["optmResponseUnitTotalLift"].iloc[0],
                    self.dt_optim_out["optmResponseUnitTotalLiftUnbound"].iloc[0],
                ],
                "total_roi": [
                    init_total_roi,
                    optm_total_roi_bounded,
                    optm_total_roi_unbounded,
                ],
                "total_cpa": [
                    init_total_cpa,
                    optm_total_cpa_bounded,
                    optm_total_cpa_unbounded,
                ],
            }
        )

        # Create df_roi (similar to R's df_roi transformation)
        df_spend = self.resp_metric[["type", "total_spend"]].rename(
            columns={"total_spend": "value"}
        )
        df_spend["name"] = "total spend"
        df_response = self.resp_metric[["type", "total_response"]].rename(
            columns={"total_response": "value"}
        )
        df_response["name"] = "total response"
        df_roi = pd.concat([df_spend, df_response])

        # Calculate normalized values (matching R's logic)
        df_roi["value_norm"] = df_roi.apply(
            lambda x: (
                x["value"]
                if self.metric == "ROAS"
                else x["value"] / df_roi[df_roi["name"] == x["name"]].iloc[0]["value"]
            ),
            axis=1,
        )

        # Create subplot titles (matching R's labs)
        subplot_titles = [
            f"Initial<br>"
            f"Spend: {self._format_num(0)}<br>"
            f"Resp: {self._format_num(0)}<br>"
            f"{self.metric}: {round(init_total_cpa if self.metric == 'CPA' else init_total_roi, 2)}",
            f"Bounded{'^' if optm_topped_bounded else ''}<br>"
            f"Spend: {self._format_num(100 * (optm_total_spend_bounded - init_total_spend) / init_total_spend)}<br>"
            f"Resp: {self._format_num(100 * self.resp_metric['total_response_lift'].iloc[1])}<br>"
            f"{self.metric}: {round(optm_total_cpa_bounded if self.metric == 'CPA' else optm_total_roi_bounded, 2)}",
            f"Bounded x{bound_mult}{'^' if optm_topped_unbounded else ''}<br>"
            f"Spend: {self._format_num(100 * (optm_total_spend_unbounded - init_total_spend) / init_total_spend)}<br>"
            f"Resp: {self._format_num(100 * self.resp_metric['total_response_lift'].iloc[2])}<br>"
            f"{self.metric}: {round(optm_total_cpa_unbounded if self.metric == 'CPA' else optm_total_roi_unbounded, 2)}",
        ]

        # Create plot with improved layout
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,  # Reduce spacing between all columns
        )

        # Define colors matching the reference
        colors = {
            "Initial": "#C0C0C0",  # Silver
            "Bounded": "#4682B4",  # Steel Blue
            f"Bounded x{bound_mult}": "#DAA520",  # Golden Rod
        }

        # Define spacing parameters (adapted for plotly)
        bar_width = 0.2  # Make bars narrower

        # Plot bars for each type
        for i, type_val in enumerate(self.resp_metric["type"]):
            type_data = df_roi[df_roi["type"] == type_val]

            # Calculate x positions for spend and response bars
            x_positions = ["total spend", "total response"]

            fig.add_trace(
                go.Bar(
                    x=type_data["name"],
                    y=type_data["value_norm"],
                    name=type_val,
                    marker_color=colors[type_val],
                    text=[
                        self._format_num(val, abbr=True) for val in type_data["value"]
                    ],
                    textposition="outside",
                    textfont=dict(size=8),
                    showlegend=True,  # Show legend for all groups
                    width=bar_width,
                    hoverinfo="none",
                ),
                row=1,
                col=i + 1,
            )

        # Update layout with improved formatting
        y_max = df_roi["value_norm"].max() * 1.2
        fig.update_layout(
            title={
                "text": f"Total Budget Optimization Result (scaled up to {self.dt_optim_out['periods'].iloc[0]})",
                "y": 0.95,
                "x": 0.02,  # Left-aligned title
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 10},
            },
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "top",  # Keep as top
                "y": 0.9,  # Higher value to position near the top
                "xanchor": "left",
                "x": 0.02,
                "font": {"size": 10},
            },
            height=500,
            width=1000,
            template="plotly_white",
            margin=dict(t=120, b=50, l=50, r=50),
            bargap=0.01,  # Reduce gap between bars within groups (make them closer)
            bargroupgap=0.5,  # Keep the gap between groups
        )

        # Update axes
        for i in range(1, 4):
            # Update y-axes
            fig.update_yaxes(
                range=[0, y_max],
                showticklabels=False,
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128, 128, 128, 0.2)",
                showline=False,  # Hide y-axis line
                row=1,
                col=i,
            )

            # Update x-axes
            fig.update_xaxes(
                tickangle=45,
                tickfont=dict(size=8),
                showline=False,  # Hide x-axis line
                row=1,
                col=i,
            )

        # Update subplot titles font size and position
        for annotation in fig["layout"]["annotations"]:
            if annotation["text"] in subplot_titles:
                annotation.update(
                    {
                        "font": dict(size=10, weight="bold"),  # Make text bold
                        "y": 1.08,  # Lower the text position
                        "yanchor": "bottom",
                        "xanchor": "center",
                        "align": "center",
                    }
                )

        return fig

    @staticmethod
    def _format_num(
        num: float,
        signif: int = 3,
        abbr: bool = False,
        pos: str = "%",
        sign: bool = True,
    ) -> str:
        """Format numbers for display."""
        if abbr:
            if abs(num) >= 1e9:
                return f"{num/1e9:.1f}B"
            if abs(num) >= 1e6:
                return f"{num/1e6:.1f}M"
            if abs(num) >= 1e3:
                return f"{num/1e3:.1f}K"
            return f"{num:.1f}"

        formatted = f"{num:.{signif}g}"
        if sign and num > 0:
            formatted = "+" + formatted
        if pos:
            formatted += pos
        return formatted

    def _plot_allocation_comparison(self) -> Dict[str, Any]:
        """Create response and spend comparison plot."""

        # Create the base dataframe for plotting
        df_plots = pd.DataFrame()

        # Response share data
        response_share = pd.DataFrame(
            {
                "channel": self.dt_optim_out["channels"],
                "Initial": self.dt_optim_out["initResponseUnitShare"],
                "Bounded": self.dt_optim_out["optmResponseUnitShare"],
                "Unbounded": self.dt_optim_out["optmResponseUnitShareUnbound"],
            }
        ).melt(id_vars=["channel"], var_name="type", value_name="response_share")

        # Spend share data
        spend_share = pd.DataFrame(
            {
                "channel": self.dt_optim_out["channels"],
                "Initial": self.dt_optim_out["initSpendShare"],
                "Bounded": self.dt_optim_out["optmSpendShareUnit"],
                "Unbounded": self.dt_optim_out["optmSpendShareUnitUnbound"],
            }
        ).melt(id_vars=["channel"], var_name="type", value_name="spend_share")

        # Mean spend data
        mean_spend = pd.DataFrame(
            {
                "channel": self.dt_optim_out["channels"],
                "Initial": self.dt_optim_out["initSpendUnit"],
                "Bounded": self.dt_optim_out["optmSpendUnit"],
                "Unbounded": self.dt_optim_out["optmSpendUnitUnbound"],
            }
        ).melt(id_vars=["channel"], var_name="type", value_name="mean_spend")

        # ROI/CPA data
        metric = "ROAS"  # or "CPA" based on your needs
        if metric == "ROAS":
            roi_cpa = pd.DataFrame(
                {
                    "channel": self.dt_optim_out["channels"],
                    "Initial": self.dt_optim_out["initRoiUnit"],
                    "Bounded": self.dt_optim_out["optmRoiUnit"],
                    "Unbounded": self.dt_optim_out["optmRoiUnitUnbound"],
                }
            ).melt(id_vars=["channel"], var_name="type", value_name="channel_roi")
        else:
            roi_cpa = pd.DataFrame(
                {
                    "channel": self.dt_optim_out["channels"],
                    "Initial": self.dt_optim_out["initCpaUnit"],
                    "Bounded": self.dt_optim_out["optmCpaUnit"],
                    "Unbounded": self.dt_optim_out["optmCpaUnitUnbound"],
                }
            ).melt(id_vars=["channel"], var_name="type", value_name="channel_cpa")

        # Marginal ROI/CPA data
        if metric == "ROAS":
            marginal = pd.DataFrame(
                {
                    "channel": self.dt_optim_out["channels"],
                    "Initial": self.dt_optim_out["initResponseMargUnit"],
                    "Bounded": self.dt_optim_out["optmResponseMargUnit"],
                    "Unbounded": self.dt_optim_out["optmResponseMargUnitUnbound"],
                }
            ).melt(id_vars=["channel"], var_name="type", value_name="marginal_roi")
        else:
            marginal = pd.DataFrame(
                {
                    "channel": self.dt_optim_out["channels"],
                    "Initial": 1 / self.dt_optim_out["initResponseMargUnit"],
                    "Bounded": 1 / self.dt_optim_out["optmResponseMargUnit"],
                    "Unbounded": 1 / self.dt_optim_out["optmResponseMargUnitUnbound"],
                }
            ).melt(id_vars=["channel"], var_name="ROAS type", value_name="marginal_cpa")

        # Combine all dataframes
        df_plots = response_share.merge(spend_share, on=["channel", "type"])
        df_plots = df_plots.merge(mean_spend, on=["channel", "type"])
        df_plots = df_plots.merge(roi_cpa, on=["channel", "type"])
        df_plots = df_plots.merge(marginal, on=["channel", "type"])

        # Update metrics to match R format with % suffix
        metrics = [
            "abs.mean\nspend",
            "mean\nspend%",
            "mean\nresponse%",
            f"mean\n{metric}",
            f"m{metric}",
        ]

        plot_data = []
        for metric_name in metrics:
            if metric_name == "abs.mean\nspend":
                values = df_plots["mean_spend"]
            elif metric_name == "mean\nspend%":  # Note the % suffix
                values = df_plots["spend_share"]
            elif metric_name == "mean\nresponse%":  # Note the % suffix
                values = df_plots["response_share"]
            elif metric_name.startswith("mean\n"):
                values = df_plots[
                    "channel_roi" if "ROAS" in metric_name else "channel_cpa"
                ]
            else:  # mROAS/mCPA calculation
                values = df_plots[
                    "marginal_roi" if "ROAS" in metric_name else "marginal_cpa"
                ]

            temp_df = pd.DataFrame(
                {
                    "channel": df_plots["channel"],
                    "type": df_plots["type"],
                    "metric": metric_name,
                    "values": values,
                }
            )
            plot_data.append(temp_df)

        df_plot_share = pd.concat(plot_data)

        # Format values
        df_plot_share["values"] = df_plot_share["values"].fillna(0)
        df_plot_share["values"] = df_plot_share["values"].replace([np.inf, -np.inf], 0)
        df_plot_share["values"] = df_plot_share["values"].clip(upper=1e15)

        # Create labels
        df_plot_share["values_label"] = df_plot_share.apply(
            lambda x: (
                f"{x['values']:,.1f}"
                if x["metric"] == "abs.mean\nspend"
                else (
                    f"{x['values']*100:.1f}%"
                    if x["metric"] in ["mean\nspend%", "mean\nresponse%"]
                    else f"{x['values']:.2f}"
                )
            ),
            axis=1,
        )

        # Start the plotting changes here
        try:
            # Create figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(10, 6))

            # Define color schemes for each type
            color_schemes = {
                "Initial": sns.light_palette("#C0C0C0", as_cmap=True),
                "Bounded": sns.light_palette("#6495ED", as_cmap=True),
                "Unbounded": sns.light_palette("#efb400", as_cmap=True),
            }

            # Plot for each type (Initial, Bounded, Unbounded)
            for i, type_name in enumerate(["Initial", "Bounded", "Unbounded"]):
                type_data = df_plot_share[df_plot_share["type"] == type_name]

                # Create pivot table with correct metric order
                plot_data = type_data.pivot(
                    index="channel", columns="metric", values="values"
                )[
                    metrics
                ]  # Explicitly specify metric order

                # Format display values
                display_data = plot_data.copy()
                display_data["abs.mean\nspend"] = display_data["abs.mean\nspend"].apply(
                    lambda x: f"{x/1000:.0f}K"
                )
                for col in metrics[1:]:
                    display_data[col] = display_data[col].apply(
                        lambda x: (f"{x*100:.1f}%" if col.endswith("%") else f"{x:.2f}")
                    )

                # Normalize data for color intensity
                norm_data = plot_data.copy()
                for col in metrics:
                    col_values = norm_data[col].values
                    if col_values.any():
                        min_val = col_values.min()
                        max_val = col_values.max()
                        if max_val > min_val:
                            norm_data[col] = (col_values - min_val) / (
                                max_val - min_val
                            )
                        else:
                            norm_data[col] = 0

                # Create heatmap
                sns.heatmap(
                    norm_data,
                    ax=axes[i],
                    cmap=color_schemes[type_name],
                    annot=display_data.values,
                    fmt="",
                    cbar=False,
                    annot_kws={"fontsize": 8},
                )

                # Customize axis
                axes[i].set_title(type_name, fontsize=8)
                axes[i].set_ylabel("Paid Media" if i == 0 else "", fontsize=8)
                axes[i].tick_params(axis="both", labelsize=8)
                axes[i].set_xlabel("")  # Remove the "metric" label

                # Rotate x-axis labels
                axes[i].set_xticklabels(
                    axes[i].get_xticklabels(), rotation=45, horizontalalignment="right"
                )

            plt.suptitle(
                f"Budget Allocation per Paid Media Variable per {self.budget_allocator.mmm_data.mmmdata_spec.interval_type}*",
            )
            plt.tight_layout(pad=2.0)

            # Add this line to prevent double display
            plt.close(fig)

            return fig

        except Exception as e:
            self.logger.error(
                "Failed to create response spend comparison plot: %s", str(e)
            )
            raise

    def _plot_response_curves(self):
        """Create response curves plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 1. Create constraint labels
        constr_labels = []
        for _, row in self.dt_optim_out.iterrows():
            label = (
                f"{row['channels']}\n"
                f"[{row['constr_low']} - {row['constr_up']}] & "
                f"[{round(row['constr_low_unb'], 1)} - {round(row['constr_up_unb'], 1)}]"
            )
            constr_labels.append(
                {
                    "channel": row["channels"],
                    "constr_label": label,
                    "constr_low_abs": row["constr_low_abs"],
                    "constr_up_abs": row["constr_up_abs"],
                    "constr_low_unb_abs": row["constr_low_unb_abs"],
                    "constr_up_unb_abs": row["constr_up_unb_abs"],
                }
            )
        constr_labels = pd.DataFrame(constr_labels)

        # 2. Merge plotDT_scurve with constraint labels
        plotDT_scurve = self.eval_list["plotDT_scurve"].merge(
            constr_labels, on="channel"
        )

        # 3. Process mainPoints data
        mainPoints = self.eval_list["mainPoints"].merge(constr_labels, on="channel")
        mainPoints = mainPoints.merge(
            self.resp_metric[["type", "type_lab"]], on="type", how="left"
        )

        # Handle type column first (matching R's mutate)
        mainPoints["type"] = mainPoints["type"].astype(str)
        mainPoints["type"] = pd.Categorical(
            mainPoints["type"].fillna("Carryover"),
            categories=["Carryover"] + list(self.resp_metric["type"].unique()),
        )

        # Handle type_lab column (matching R's mutate)
        mainPoints["type_lab"] = mainPoints["type_lab"].astype(str)
        mainPoints["type_lab"] = pd.Categorical(
            mainPoints["type_lab"].fillna("Carryover"),
            categories=["Carryover"] + list(self.resp_metric["type_lab"].unique()),
        )
        # Get carryover points
        caov_points = mainPoints[mainPoints["type"] == "Carryover"][
            ["channel", "spend_point"]
        ].rename(columns={"spend_point": "caov_spend"})

        # Merge and calculate constraint bounds
        mainPoints = mainPoints.merge(caov_points, on="channel")

        # Get the levels from resp_metric
        levs1 = self.resp_metric[
            "type"
        ].unique()  # Should contain ["Initial", "Unbounded"]

        # Calculate constraint bounds directly using pandas operations
        mainPoints["constr_low_abs"] = np.where(
            mainPoints["type"] == levs1[1],  # levs1[1] should be "Initial"
            mainPoints["constr_low_abs"] + mainPoints["caov_spend"],
            np.nan,
        )
        mainPoints["constr_up_abs"] = np.where(
            mainPoints["type"] == levs1[1],
            mainPoints["constr_up_abs"] + mainPoints["caov_spend"],
            np.nan,
        )
        mainPoints["constr_low_unb_abs"] = np.where(
            mainPoints["type"] == levs1[2],  # levs1[2] should be "Unbounded"
            mainPoints["constr_low_unb_abs"] + mainPoints["caov_spend"],
            np.nan,
        )
        mainPoints["constr_up_unb_abs"] = np.where(
            mainPoints["type"] == levs1[2],
            mainPoints["constr_up_unb_abs"] + mainPoints["caov_spend"],
            np.nan,
        )

        # Calculate plot bounds
        mainPoints["plot_lb"] = mainPoints["constr_low_abs"].fillna(
            mainPoints["constr_low_unb_abs"]
        )
        mainPoints["plot_ub"] = mainPoints["constr_up_abs"].fillna(
            mainPoints["constr_up_unb_abs"]
        )

        # 4. Create the plot with improved layout
        num_channels = len(plotDT_scurve["constr_label"].unique())
        num_rows = (num_channels + 2) // 3

        fig = make_subplots(
            rows=num_rows,
            cols=3,
            subplot_titles=plotDT_scurve["constr_label"].unique(),
            horizontal_spacing=0.15,  # Increased from 0.1
            vertical_spacing=0.25,  # Increased from 0.15
        )

        # Add traces for each channel
        for i, channel in enumerate(plotDT_scurve["constr_label"].unique()):
            row = (i // 3) + 1
            col = (i % 3) + 1

            channel_data = plotDT_scurve[plotDT_scurve["constr_label"] == channel]
            channel_points = mainPoints[mainPoints["constr_label"] == channel]

            # Carryover area first
            carryover_data = channel_data[
                channel_data["spend"] <= channel_data["mean_carryover"].iloc[0]
            ]
            if not carryover_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=carryover_data["spend"],
                        y=carryover_data["total_response"],
                        fill="tozeroy",
                        fillcolor="rgba(128, 128, 128, 0.4)",
                        mode="none",  # Changed from line=dict(width=0)
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Response curve second
            fig.add_trace(
                go.Scatter(
                    x=channel_data["spend"],
                    y=channel_data["total_response"],
                    mode="lines",
                    name=channel,
                    line=dict(width=0.5),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add points and error bars
            if not channel_points.empty:
                # Get the bound multiplier from the data
                bound_mult = self.dt_optim_out["unconstr_mult"].iloc[0]

                # Create color mapping dictionary with dynamic bounded multiplier
                color_map = {
                    "Carryover": "white",
                    "Initial": "grey",
                    "Bounded": "steelblue",
                    f"Bounded x{bound_mult}": "darkgoldenrod",
                }

                # Add points for each type (only add to legend for first subplot)
                for type_label in color_map.keys():
                    # Use type column for Carryover, type_lab for others
                    if type_label == "Carryover":
                        type_points = channel_points[
                            channel_points["type"] == "Carryover"
                        ]
                    else:
                        type_points = channel_points[
                            channel_points["type_lab"] == type_label
                        ]

                    if not type_points.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=type_points["spend_point"],
                                y=type_points["response_point"],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color=color_map[type_label],
                                    line=dict(color="black", width=1),
                                ),
                                name=type_label,
                                legendgroup=type_label,
                                showlegend=(
                                    i == 0
                                ),  # Only show in legend for first subplot
                            ),
                            row=row,
                            col=col,
                        )

                # Add error bars only for Bounded and Bounded x{bound_mult} points
                bounded_points = channel_points[
                    channel_points["type_lab"].isin(
                        ["Bounded", f"Bounded x{bound_mult}"]
                    )
                ].copy()  # Add .copy() to avoid SettingWithCopyWarning

                if not bounded_points.empty:
                    # First add the dotted lines between bounds
                    for _, point in bounded_points.iterrows():
                        if pd.notna(point["plot_lb"]) and pd.notna(point["plot_ub"]):
                            fig.add_trace(
                                go.Scatter(
                                    x=[point["plot_lb"], point["plot_ub"]],
                                    y=[
                                        point["response_point"],
                                        point["response_point"],
                                    ],
                                    mode="lines",
                                    line=dict(color="black", width=1, dash="dot"),
                                    showlegend=False,
                                ),
                                row=row,
                                col=col,
                            )

                    # Then add the triangular markers at the bounds
                    for bound, symbol in [
                        ("plot_lb", "triangle-left"),
                        ("plot_ub", "triangle-right"),
                    ]:
                        bound_points = bounded_points[pd.notna(bounded_points[bound])]
                        if not bound_points.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=bound_points[bound],
                                    y=bound_points["response_point"],
                                    mode="markers",
                                    marker=dict(
                                        symbol=symbol,
                                        size=8,
                                        color="black",
                                    ),
                                    showlegend=False,
                                ),
                                row=row,
                                col=col,
                            )

        # Update layout with improved formatting
        fig.update_layout(
            title={
                "text": (
                    f"Simulated Response Curves<br>"
                    f"<span style='font-size:10px'>"
                    f"Spend per {self.budget_allocator.mmm_data.mmmdata_spec.interval_type} "
                    f"(grey area: mean historical carryover) | "
                    f"Response [{self.budget_allocator.mmm_data.mmmdata_spec.dep_var_type}]"
                    "</span>"
                ),
                "y": 0.95,
                "x": 0.02,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 12},
            },
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.85,
                xanchor="left",
                x=1.05,
                font=dict(size=10),
            ),
            height=300 * num_rows,
            width=1000,
            template="plotly_white",
            margin=dict(t=80, b=80, l=120, r=50),  # Reduced top margin from 120 to 80
            annotations=[
                dict(
                    text=f"Spend** per {self.budget_allocator.mmm_data.mmmdata_spec.interval_type}",
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=10),
                ),
                dict(
                    text=f"Total Response [{self.budget_allocator.mmm_data.mmmdata_spec.dep_var_type}]",
                    x=-0.08,  # Adjusted position
                    y=0.35,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    textangle=-90,
                    font=dict(size=10),
                ),
            ],
        )

        # Update axes without titles (remove title_text from both)
        fig.update_xaxes(
            tickfont={"size": 8},
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(128, 128, 128, 0.2)",
        )

        fig.update_yaxes(
            tickfont={"size": 8},
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(128, 128, 128, 0.2)",
        )

        return fig

    def plot_all(
        self,
        display_plots: bool = True,
        export_location: Union[str, Path] = None,
        quiet: bool = True,
    ) -> Dict[str, plt.Figure]:
        """
        Create all allocator plots.
        Parameters:
            display_plots (bool): Whether to display the plots
            export_location (Union[str, Path]): Location to export plots
            quiet (bool): If True, suppresses logging output
        """

        try:
            plots = {
                "budget_opt": self._plot_response_spend_comparison(),
                "allocation": self._plot_allocation_comparison(),
                "response": self._plot_response_curves(),
            }

            if display_plots:
                self.display_plots(plots)

            if export_location is not None:
                self.export_plots_fig(export_location, plots)

            return plots

        except Exception as e:
            logger.error("Failed to generate all plots: %s", str(e))
            raise
