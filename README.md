# Robyn MMM Open Source Project 2.0

2021-03-03

## How To Start

There are three .R script files:

- fb_robyn.exec.R # you only need this script to execute, it calls the
  other 2 scripts
- fb_robyn.func.R # this contains feature engineering, modelling functions and plotting
- fb_robyn.optm.R # this contains the budget allocator and plotting

Two .csv files:

- de_simulated_data.csv # this is our simulated data set
- generated_holidays.csv # this contains holidays of all countries from the
  library prophet

All files should be placed in the same folder

Please make sure you've installed all library specified in
fb_robyn.exec.R first

Test run:
After installing all libraries, if you select all and run in
fb_robyn.exec.R, the script should run through and save some plots on your selected folder

## Usage Guidelines

- Latest script usage guideline: Please see comments in scripts within the source code in fb_robyn.exec.R 
- Guidelines on the website to be updated soon: https://facebookexperimental.github.io/Robyn/docs/step-by-step-guide

## Model selection with evolutionary algorithm

Using Facebook AI's open source gradient-free optimisation library [Nevergrad](https://facebookresearch.github.io/nevergrad/), Robyn is able to leverage evolutionary algorithms to perform multi-objective hyperparameter optimisation and output a set of Pareto-optimal solutions. Besides NRMSE as loss function for the optimisation, Robyn also minimises on a business logic "decomposition distance", or DECOMP.RSSD that is aiming to steer the model towards more realistic decomposition results. In case of calibration, a third loss function MAPE.LIFT is added too.

The following plot demonstrates typical Pareto fronts 1-3 on NRMSE and DECOMP.RSSD:
![paretofront](https://user-images.githubusercontent.com/14415136/110000483-a3269f00-7d13-11eb-85de-0bae918f4f5c.png)


## Join the FB Robyn MMM community. **Coming soon**

## FB Contact

- gufeng@fb.com, Gufeng Zhou, Marketing Science Partner
- leonelsentana@fb.com, Leonel Sentana, Marketing Science Partner
- aprada@fb.com, Antonio Prada, Marketing Science Partner
- igorskokan@fb.com, Igor Skokan, Marketing Science Partner

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

FB Robyn MMM R script is MIT licensed, as found in the LICENSE file.

- Terms of Use - https://opensource.facebook.com/legal/terms 
- Privacy Policy - https://opensource.facebook.com/legal/privacy
