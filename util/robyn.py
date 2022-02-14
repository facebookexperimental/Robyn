# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS

import rpy2

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')


########################################################################################################################
# RESEARCH
# Remove later - 2022.02.14

# Overview documentation
# https://rpy2.github.io/doc/latest/html/overview.html#
# https://rpy2.github.io/doc/v2.9.x/html/introduction.html

# Installation instructions
# https://rpy2.github.io/doc/latest/html/overview.html#install-installation
# conda install -c r rpy2  # did not work
# conda install -c conda-forge rpy2

# Check location
print(f'ryp2 location: {rpy2.__path__}')

# See ryp2 version
print(f'ryp2 version: {rpy2.__version__}')


########################################################################################################################
# Step 1: Setup Environment

# Import rpy2's package module
import rpy2.robjects.packages as rpackages

# Import R's utility package
utils = rpackages.importr('utils')

# Select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list


# INSTALL NEEDED PACKAGES
# R package names
packnames = ['Robyn']

# R vector of strings
from rpy2.robjects.vectors import StrVector

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))




###############
# dict_params = {
#     'dt_input': dt_simulated_weekly  # pandas data frame with dates
# }
#
# print(f'Hey, I just want you to see this list; {l_rand}')
#
#
# obj_made = Robyn(dt_obj=,
#                  )