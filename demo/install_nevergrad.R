# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#### Nevergrad installation guide
## ATTENTION: Python version 3.10+ version may cause Nevergrad error
## See here for more info about installing Python packages via reticulate
## https://rstudio.github.io/reticulate/articles/python_packages.html

# Install reticulate first if you haven't already
install.packages("reticulate")

#### Option 1: nevergrad installation via PIP
# 1. load reticulate
library("reticulate")
# 2. create virtual environment
virtualenv_create("r-reticulate")
# 3. use the environment created
use_virtualenv("r-reticulate", required = TRUE)
# 4. point Python path to the python file in the virtual environment. Below is
#    an example for MacOS M1 or above. The "~" is my home dir "/Users/gufengzhou".
#    Show hidden files in case you want to locate the file yourself.
Sys.setenv(RETICULATE_PYTHON = "~/.virtualenvs/r-reticulate/bin/python")
# 5. Check python path
py_config() # If the first path is not as 4, do 6
# 6. Restart R session, run #4 first, then load library("reticulate"), check
#    py_config() again, python should have path as in #4.
#    If you see: "NOTE: Python version was forced by RETICULATE_PYTHON_FALLBACK"
#    if you're using RStudio, go to Global Options > Python, and uncheck the
#    box for "Automatically activate project-local Python environments".
# 7. Install numpy if py_config shows it's not available
py_install("numpy", pip = TRUE)
# 8. Install nevergrad
py_install("nevergrad", pip = TRUE)
# 9. If successful, py_config() should show numpy and nevergrad with installed paths
# 10. Everytime R session is restarted, you need to run #4 first to assign python
#    path before loading Robyn
# 11. Alternatively, add the line RETICULATE_PYTHON = "~/.virtualenvs/r-reticulate/bin/python"
#    in the file Renviron in the the R directory to force R to always use this path by
#    default. Run R.home() to check R path, then look for file "Renviron". This way, you
#    don't need to run #4 everytime. Note that forcing this path impacts other R instances
#    that requires different Python versions

#### Option 2: nevergrad installation via conda
# 1. load reticulate
library("reticulate")
# 2. Install conda if not available
install_miniconda()
# 3. create virtual environment
conda_create("r-reticulate")
# 4. use the environment created
use_condaenv("r-reticulate")
# 5. point Python path to the python file in the virtual environment. Below is
#    an example for MacOS M1 or above. The "~" is my home dir "/Users/gufengzhou".
#    Show hidden files in case you want to locate the file yourself
Sys.setenv(RETICULATE_PYTHON = "~/Library/r-miniconda-arm64/envs/r-reticulate/bin/python")
# 6. Check python path
py_config() # If the first path is not as 5, do 7
# 7. Restart R session, run #5 first, then load library("reticulate"), check
#    py_config() again, python should have path as in #5
# 8. Install numpy if py_config shows it's not available
conda_install("r-reticulate", "numpy", pip=TRUE)
# 9. Install nevergrad
conda_install("r-reticulate", "nevergrad", pip=TRUE)
# 10. If successful, py_config() should show numpy and nevergrad with installed paths
# 11. Everytime R session is restarted, you need to run #4 first to assign python
#    path before loading Robyn
# 12. Alternatively, add the line RETICULATE_PYTHON = "~/Library/r-miniconda-arm64/envs/r-reticulate/bin/python"
#    in the file Renviron in the the R directory to force R to always use this path by
#    default. Run R.home() to check R path, then look for file "Renviron". This way, you
#    don't need to run #5 everytime. Note that forcing this path impacts other R instances
#    that requires different Python versions

#### Known potential issues when installing nevergrad and possible fixes
# Fix SSL issue: reticulate:::rm_all_reticulate_state()
# Try updating pip: system("pip3 install --upgrade pip")
# Check this issue for more ideas to debug your reticulate/nevergrad issues:
# https://github.com/facebookexperimental/Robyn/issues/189
