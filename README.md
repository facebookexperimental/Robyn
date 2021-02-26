# FB NextGen MMM beta R script, v19

2020-04-29

###### How To Start

The following files are contained in the zip:

Four .R script files:

- fb_nextgen_mmm_v19.exec.R # you only need this script to execute, it calls the
  other 3 scripts
- fb_nextgen_mmm_v19.func.R # this contains all major functions and data
  preparation
- fb_nextgen_mmm_v19.plot.R # this contains all plot functions
- fb_nextgen_mmm_v19.bayes.R # this contains adapted functions from the library
  parBayesianOptimization

Two .csv files:

- de_simulated_data.csv # this is our simulated data set
- generated_holidays.csv # this contains holidays of all countries from the
  library prophet

All files should be placed in the same folder

Please make sure you've installed all library specified in
fb_nextgen_mmm_v19.exec.R first

Initial setting is running 1000 random search trails with geometric adstockinng.
After installing all libraries, if you select all and run in
fb_nextgen_mmm_v19.exec.R, the script should run and produce one plot

Script execution details please see comments in scripts

## Join the FB NextGen MMM community

###### FB Contact

gufeng@fb.com, Gufeng Zhou, Marketing Science Partner leonelsentana@fb.com,
Leonel Sentana, Marketing Science Partner aprada@fb.com, Antonio Prada,
Marketing Science Partner igorskokan@fb.com, Igor Skokan, Marketing Science
Partner

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

FB NextGen MMM R script is MIT licensed, as found in the LICENSE file.

Terms of Use - https://opensource.facebook.com/legal/terms Privacy Policy -
https://opensource.facebook.com/legal/privacy
