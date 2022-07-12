# Setup R Environment
# https://facebookexperimental.github.io/Robyn/docs/quick-start/

install.packages('remotes')
remotes::install_github("facebookexperimental/Robyn/R")

library(Robyn)
library(reticulate)


.libPaths()

Sys.which("R")
echo $PATH


# old.packages()