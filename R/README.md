# Project Robyn - Continuous & Semi-Automated MMM <img src='man/figures/logo.png' align="right" height="139px" />
### The Open Source Marketing Mix Model Package from Facebook Marketing Science

## R Quick Start
  
  * Install Robyn latest package version:
```{r}
## CRAN VERSION: TBA
# install.packages("Robyn")

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

FB Robyn MMM R script is MIT licensed, as found in the LICENSE file.

- Terms of Use - https://opensource.facebook.com/legal/terms 
- Privacy Policy - https://opensource.facebook.com/legal/privacy

## Contact

* gufeng@fb.com, Gufeng Zhou, Marketing Science Partner
* leonelsentana@fb.com, Leonel Sentana, Marketing Science Partner
* igorskokan@fb.com, Igor Skokan, Marketing Science Partner
* bernardolares@fb.com, Bernardo Lares, Marketing Science Partner
* kylegoldberg@fb.com, Kyle Goldberg, Marketing Science Partner
