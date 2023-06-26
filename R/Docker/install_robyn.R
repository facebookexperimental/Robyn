# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#Sys.setenv(DOWNLOAD_STATIC_LIBV8=1)

install.packages("remotes")

install.packages("reticulate")

remotes::install_github("facebookexperimental/Robyn/R")


library(reticulate)

#Sys.setenv(RETICULATE_PYTHON = "/usr/bin/python3.8")

Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin")



print(" ---------- PYTHON PATH IN RSESSION:")

print(Sys.which("python"))

print(reticulate::py_config())



reticulate::conda_create("r-reticulate", "Python 3.9") # Only works with <= Python 3.9 sofar

reticulate::use_condaenv("r-reticulate")

reticulate::conda_install("r-reticulate", "nevergrad", pip=TRUE)



#virtualenv_create("r-reticulate")

#virtualenv_exists("r-reticulate")

#sessionInfo()
