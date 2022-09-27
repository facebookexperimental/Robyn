# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(paste(
    ">>> Welcome to RobynLearn. To load the interactive app, run: robyn_learn()"
  ))
}
