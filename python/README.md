# Robyn: Continuous & Semi-Automated MMM <img src='R/man/figures/logo.png' align="right" height="139px" />
### The Open Source Marketing Mix Model Package from Meta Marketing Science

<!-- [![Pypi\_Status\_Badge](https://www.r-pkg.org/badges/version/Robyn)](https://cran.r-project.org/package=Robyn) [![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/Robyn?color=green)](https://cranlogs.r-pkg.org/badges/grand-total/Robyn?color=green) [![Site](https://img.shields.io/badge/site-Robyn-blue.svg)](https://facebookexperimental.github.io/Robyn/) [![Facebook](https://img.shields.io/badge/group-Facebook-blue.svg)](https://www.facebook.com/groups/robynmmm/) [![CodeFactor](https://www.codefactor.io/repository/github/facebookexperimental/robyn/badge)](https://www.codefactor.io/repository/github/facebookexperimental/robyn) -->
---

## Introduction

  * **What is Robyn?**: Robyn is an experimental, semi-automated and open-sourced Marketing Mix Modeling (MMM) package from Meta Marketing Science. It uses various machine learning techniques (Ridge regression, multi-objective evolutionary algorithm for hyperparameter optimization, time-series decomposition for trend & season, gradient-based optimization for budget allocation, clustering, etc.) to define media channel efficiency and effectivity, explore adstock rates and saturation curves. It's built for granular datasets with many independent variables and therefore especially suitable for digital and direct response advertisers with rich data sources. 
  
  * **Why are we doing this?**: MMM used to be a resource-intensive technique that was only affordable for "big players". As the privacy needs of the measurement landscape evolve, there's a clear trend of increasing demand for modern MMM as a privacy-safe solution. At Meta Marketing Science, our mission is to help all businesses grow by transforming marketing practices grounded in data and science. It's highly aligned with our mission to democratizing MMM and making it accessible for advertisers of all sizes. With Project Robyn, we want to contribute to the measurement landscape, inspire the industry and build a community for exchange and innovation around the future of MMM and Marketing Science in general.
  
## Quick start for Python (Beta)

  The Python version of Robyn is rewritten from Robyn's R package version `3.11.1` to Python using object oriented programming principles and modular architecture for a robust solution. It was developed by utilizing various LLMs and AI workflows. As is common with any AI-based solutions, there may be potential challenges in translating code from one language to another.
  In this case, we anticipate that there could be some issues in the translation from R to Python. However, we believe in the power of community collaboration and open-source contribution. Therefore, we are opening this project to the community to participate and contribute.
  Together, we can address and resolve any issues that may arise, enhancing the functionality and efficiency of the Python version of Robyn. We look forward to your contributions and to the continuous improvement of this project.

**1. Installing the package**
  
  * Install Robyn latest package version:
```{r}
## Pypi
pip3 install robynpy

## DEV VERSION
# if you are pulling source from github, install dependencies using requirements.txt
pip3 install -r requirements.txt
```
  
**2. Getting started**

  * python/src/robyn/tutorials contains tutorials for most common scenarios. Tutorials use simulated dataset provided in the package.

  * There are two ways of running Python Robyn; one is `tutorial1.ipynb` and second is `tutorial1_src.ipynb`.

**3. Running end-to-end**

Option 1:
  * `tutorial1.ipynb` is the main notebook that runs the end-to-end flow. It is designed for majority of the users who would prefer a one click solution that runs the robyn flow end-to-end with minimal knowledge of the underlying logic. It should run without any changes required if you wish to use the simulated dataset for testing purposes. 

  * This notebook uses APIs available in `python/src/robyn/robyn.py` to set the configs, run feature engineering, run model training, evaluate models with clustering, generate one pagers and perform budget allocation.
  
  * Change any of the configs directly in the notebook and avoid changes to robyn.py for what can be configurable.

Option 2:
  * `tutorial1_src.ipynb` runs the end-to-end flow of robyn python but with a lot more flexibility. It is designed for users who would like to have more control over which modules are and aren't run (ie. skipping clustering/one pager plots/budget allocation etc.). It should run without any changes required if you wish to use the simulated dataset for testing purposes. 

  * This notebook doesn't use APIs available in `python/src/robyn/robyn.py` but instead, calls the modules directly with the appropriate parameters. In this way, it is more flexible but still expects the users to understand the underlying logic that may change when using various parameter values.

## Helpful Links

  * Visit our [website](https://facebookexperimental.github.io/Robyn/) to explore more details about Project Robyn.
  
  * Join our [public group](https://www.facebook.com/groups/robyn/) to exchange with other users and interact with team Robyn.
  
  * Take Meta's [official Robyn blueprint course](https://www.facebookblueprint.com/student/path/253121-marketing-mix-models?utm_source=readme) online 

## License

Meta's Robyn is MIT licensed, as found in the LICENSE file.

- Terms of Use - https://opensource.facebook.com/legal/terms 
- Privacy Policy - https://opensource.facebook.com/legal/privacy
- Defensive Publication - https://www.tdcommons.org/dpubs_series/4627/

## Contact

* gufeng@meta.com, Gufeng Zhou, Marketing Science, Robyn creator
* igorskokan@meta.com, Igor Skokan, Marketing Science Director, open source
