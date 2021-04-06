---
id: automated-hyperparameter-selection-optimization
title: Automated hyperparameter selection and optimization
---

import useBaseUrl from '@docusaurus/useBaseUrl';

MMMs are likely to **contain high cardinality of parameters**. ie. alphas and gammas for the diminishing returns (Hill) function, as well as, thetas for geometric ad stock transformation.
In addition, parameters dimensionality increases proportionally with the total number of marketing channels to be measured. Thus, it is extremely necessary to deal with a high dimensionality parameter space where, the greater the number of parameters, **the greater the model complexity, its dimensionality and computational requirements.**

In order to achieve computational efficiency while optimizing overall model accuracy, we leverage [**Facebook’s Nevergrad gradient-free optimization platform**](https://facebookresearch.github.io/nevergrad/).  Nevergrad allows us to optimize the explore and exploit balance through the **ask** and **tell** commands, in order to perform a multi-objective optimization tha balances out the Normalized Root Mean Square Error (**NRMSE**) and **decomp.RSSD** ratio (Relationship between spend share and channels coefficient decomposition share) providing a set of **Pareto optimal model solutions**

Please find below an example of a common chart for the Pareto model solutions. Each dot in the chart represents an explored model solution, while the first three lines in the lower-left corner represent the best model solutions.

<img alt="pareto chart" src={useBaseUrl('/img/pareto1.png')} />

The premise of an **evolutionary algorithm** is that of natural selection. In an EA you may have a set of iterations where some combinations of coefficients that will be explored by the model will survive and proliferate, while unfit models will die off and not contribute to the gene pool of further generations, much like in natural selection. In robyn, we recommend a minimum of 500 iterations where each of these will provide feedback to its upcoming generation, and therefore guide the model towards the optimal coefficient values for alphas, gammas and thetas. We also recommend a minimum of 40 trials which are a set of independent initiations of the model that will each of them have the number of iterations you set under ‘set_iter’ object. E.g. 500 iterations on set_iter x 40 trials = 20000 different iterations and possible model solutions.
