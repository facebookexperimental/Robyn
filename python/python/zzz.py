# (c) Meta Platforms, Inc. and affiliates.

import reticulate

## Not needed since reticulate is used to interface with Python from R


def on_load(libname, pkgname):
    reticulate.configure_environment(pkgname)
