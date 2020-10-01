---
id: doc7
title: Automated hyperparameter selection and optimization
---
MMMs are likely to contain high cardinality of parameters, ie. alphas, thetas and gammas for geometric ad stock transformation. In addition, parameters dimensionality increases proportionally with the total number of marketing channels to be measured. Thus, it is extremely necessary to deal with a high dimensionality parameter space where, the greater the number of parameters, the greater the model complexity and computational requirements. In order to achieve computational efficiency while maintaining overall model accuracy, we utilize a set of techniques and algorithms:
- A **sampling method** to search for candidates that accurately represents the multi-dimensional hyperparameter space (LHS Latinhypercube sampling)
```
#### Define latin hypercube sampling function
f.hypSamLHS <- function(set_mediaVarName, iterN, hyperbound.global, adstock)

```

- A **cross-validation scheme** that allows to have multiple dataset folds to train and validate the model. This allows using 100% of the train/validation dataset while reducing the chance of the model overfitting and defining optimal regularization lambda.
```
### fit ridge regression with x-validation
      cvmod <- cv.glmnet(x_train
                         ,y_train
                         ,family = "gaussian"
                         ,alpha = 0 #0 for ridge regression
                         ,lambda = lambda_seq
                         ,lower.limits = lower.limits
                         ,upper.limits = upper.limits
                         ,type.measure = "mse"
      )

```

- An **optimization algorithm** (Random Search) with a score function (R2or MAPE) to minimize or maximize. This allows the model to obtain optimal results that minimize the error for example, within the hyperparameter sampled space and reduce computational effort.
```
f.mmm.RSoptim <- function(hyperbound.global = hyperbound.global
                          ,iterN = iterN
                          ,setCores = setCores
                          ,epochN = Inf
                          ,out = F
                          ,temp.csv.path
)
```

- A **boosting loop** that considers previous Random Search optimization results for best hyperparameters bounds and feeds these into new Random Search optimization epochs.
```
 # hyper optimisation loop
  optim.loop <- T
  optim.iter <- 1
  if (!exists("resultRS")) {
    epoch.iter <- 1
    optimParRS.collect <- list()
  } else {
    epoch.iter <- epoch.iter + 1
    optimParRS.collect <- list(resultRS[["optimParRS"]])
  } 
  
  while (optim.loop & optim.iter <= epochN) {
    
    assign("epoch.iter", epoch.iter, envir = .GlobalEnv)
    assign("optim.iter", optim.iter, envir = .GlobalEnv)
    # run RS model with adapted 
    
    sysTimeRS <- system.time({
      resultRS <- f.mmm(hyperbound.global
                        ,iterRS = iterN
                        ,setCores = setCores
                        ,out = out
      )})
```