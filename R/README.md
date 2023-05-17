# Robyn: Continuous & Semi-Automated MMM <img src='man/figures/logo.png' align="right" height="139px" />
### The Open Source Marketing Mix Model Package from Meta Marketing Science

[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/Robyn)](https://cran.r-project.org/package=Robyn) [![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/Robyn?color=green)](https://cranlogs.r-pkg.org/badges/grand-total/Robyn?color=green) [![Site](https://img.shields.io/badge/site-Robyn-blue.svg)](https://facebookexperimental.github.io/Robyn/) [![Facebook](https://img.shields.io/badge/group-Facebook-blue.svg)](https://www.facebook.com/groups/robynmmm/) [![CodeFactor](https://www.codefactor.io/repository/github/facebookexperimental/robyn/badge)](https://www.codefactor.io/repository/github/facebookexperimental/robyn)
---

Project Robyn aims to radically change Marketing Mix Modelling (MMM) practice by utilizing automation, ground-truth calibration, and incorporating AI in an open-source approach. While demystifying the use of these techniques, Robyn helps to reduce and minimize human bias in MMM, ensuring that the results are based on objective and validated statistical analysis. The AI algorithms & ground-truth calibration used in Robyn aim to minimize the potential for subjective interpretations and decision making. This leads to a more accurate representation of the impact of marketing investments, enabling organizations of all sizes to make impartial data-driven decisions. Project Robyn, along with strong community support, represents a step towards the future of MMM, where the use of AI and automation will become the norm, providing unbiased and reliable results every time. By creating an open-source platform, Project Robyn is democratizing access to underlying technology, enabling organizations of all sizes to benefit from unbiased MMM analysis either as final end-users or agencies & consultancies building their own businesses upon Robyn. The data science, the code, and various business case studies are available for everyone to learn from and use.

## R Quick Start
  
  * Install Robyn latest package version:
```{r}
## CRAN VERSION
install.packages("Robyn")

## DEV VERSION
# If you don't have remotes installed yet, first run: install.packages("remotes")
remotes::install_github("facebookexperimental/Robyn/R")
```

  * If it's taking too long to download, you have a slow or unstable internet connection, and have [issues](https://github.com/facebookexperimental/Robyn/issues/309) while installing the package, try setting `options(timeout=400)`.
  
  * Robyn requires the Python library [Nevergrad](https://facebookresearch.github.io/nevergrad/). If encountering Python-related 
  error during installation, please check out the [step-by-step guide](https://github.com/facebookexperimental/Robyn/blob/main/demo/demo.R) to get more info.
  
  * For Windows, if you get openssl error, please see instructions
  [here](https://stackoverflow.com/questions/54558389/how-to-solve-this-error-while-installing-python-packages-in-rstudio/54566647) and
  [here](https://dev.to/danilovieira/installing-openssl-on-windows-and-adding-to-path-3mbf) to install and update openssl.

## License

Meta's Robyn is MIT licensed, as found in the LICENSE file.

- Terms of Use - https://opensource.facebook.com/legal/terms 
- Privacy Policy - https://opensource.facebook.com/legal/privacy

## Contact

* gufeng@meta.com, Gufeng Zhou, Marketing Science Partner
* leonelsentana@meta.com, Leonel Sentana, Marketing Science Partner
* igorskokan@meta.com, Igor Skokan, Marketing Science Partner
* bernardolares@meta.com, Bernardo Lares, Marketing Science Partner
