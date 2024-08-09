# Robyn: Continuous & Semi-Automated MMM <img src='R/man/figures/logo.png' align="right" height="139px" />
### The Open Source Marketing Mix Model Package from Meta Marketing Science

<!-- [![Pypi\_Status\_Badge](https://www.r-pkg.org/badges/version/Robyn)](https://cran.r-project.org/package=Robyn) [![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/Robyn?color=green)](https://cranlogs.r-pkg.org/badges/grand-total/Robyn?color=green) [![Site](https://img.shields.io/badge/site-Robyn-blue.svg)](https://facebookexperimental.github.io/Robyn/) [![Facebook](https://img.shields.io/badge/group-Facebook-blue.svg)](https://www.facebook.com/groups/robynmmm/) [![CodeFactor](https://www.codefactor.io/repository/github/facebookexperimental/robyn/badge)](https://www.codefactor.io/repository/github/facebookexperimental/robyn) -->
---

## Introduction

  * **What is Robyn?**: Robyn is an experimental, semi-automated and open-sourced Marketing Mix Modeling (MMM) package from Meta Marketing Science. It uses various machine learning techniques (Ridge regression, multi-objective evolutionary algorithm for hyperparameter optimization, time-series decomposition for trend & season, gradient-based optimization for budget allocation, clustering, etc.) to define media channel efficiency and effectivity, explore adstock rates and saturation curves. It's built for granular datasets with many independent variables and therefore especially suitable for digital and direct response advertisers with rich data sources. 
  
  * **Why are we doing this?**: MMM used to be a resource-intensive technique that was only affordable for "big players". As the privacy needs of the measurement landscape evolve, there's a clear trend of increasing demand for modern MMM as a privacy-safe solution. At Meta Marketing Science, our mission is to help all businesses grow by transforming marketing practices grounded in data and science. It's highly aligned with our mission to democratizing MMM and making it accessible for advertisers of all sizes. With Project Robyn, we want to contribute to the measurement landscape, inspire the industry and build a community for exchange and innovation around the future of MMM and Marketing Science in general.
  
## Quick start for Python (Beta)

  The Python version of Robyn was developed by utilizing Code Llama's capabilities to port code from R to Python. As is common with any AI-based solutions, there may be potential challenges in translating code from one language to another.
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

  * Use this [demo.py](https://github.com/facebookexperimental/Robyn/tree/main/python/src/scripts/demo.py) script as step-by-step guide that is
  intended to cover most common use-cases. Test the package using simulated dataset provided in the package. 
  
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
