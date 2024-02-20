# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import reticulate

## Not needed since reticulate is used to interface with Python from R


def on_load(libname, pkgname):
    reticulate.configure_environment(pkgname)
