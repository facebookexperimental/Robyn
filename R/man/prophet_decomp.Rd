% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/inputs.R
\name{prophet_decomp}
\alias{prophet_decomp}
\title{Conduct prophet decomposition}
\usage{
prophet_decomp(
  dt_transform,
  dt_holidays,
  prophet_country,
  prophet_vars,
  prophet_signs,
  factor_vars,
  context_vars,
  organic_vars,
  paid_media_spends,
  paid_media_vars,
  intervalType,
  dayInterval,
  custom_params
)
}
\arguments{
\item{dt_transform}{A data.frame with all model features.
Must contain \code{ds} column for time variable values and
\code{dep_var} column for dependent variable values.}

\item{dt_holidays}{data.frame. Raw input holiday data. Load standard
Prophet holidays using \code{data("dt_prophet_holidays")}}

\item{context_vars, paid_media_spends, intervalType, dayInterval, prophet_country, prophet_vars, prophet_signs, factor_vars}{As included in \code{InputCollect}}

\item{organic_vars}{Character vector. Typically newsletter sendings,
push-notifications, social media posts etc. Compared to \code{paid_media_vars}
\code{organic_vars} are often marketing activities without clear spends.}

\item{paid_media_vars}{Character vector. Names of the paid media variables'
exposure level metrics (impressions, clicks, GRP etc) other than spend.
The values on each of these variables must be numeric. These variables are not
being used to train the model but to check relationship and recommend to
split media channels into sub-channels (e.g. fb_retargeting, fb_prospecting,
etc.) to gain more variance. \code{paid_media_vars} must have same
order and length as \code{paid_media_spends} respectively and is not required.}

\item{custom_params}{List. Custom parameters passed to \code{prophet()}}
}
\value{
A list containing all prophet decomposition output.
}
\description{
When \code{prophet_vars} in \code{robyn_inputs()} is specified, this
function decomposes trend, season, holiday and weekday from the
dependent variable.
}
