# bayesify: Bayesian performance-influence modelling for configurable software systems

## What is it?
bayesify takes the pain out of Bayesian performance-influence modeling.
It can be used like a linear regression to determine the influence of individual software configuration options and to predict the performance of unseen configurations.
Bayesify draws from a scientific paper (see below) as it adopts the major pre-processing and modeling steps.
However, Bayesify uses [numpyro](https://github.com/pyro-ppl/numpyro) as its Probabilistic Programming backend, greatly accelerating model fitting. As a result, it may be used for new experiments, but cannot reproduce the original paper results. For a replication package, please refer to the [original supplementary website](https://github.com/AI-4-SE/Mastering-Uncertainty-in-Performance-Estimations-of-Configurable-Software-Systems).
## Main features
* probability distribution for each configuration option
* confidence intervals of custom confidence for the influence of each configuration option
* probability distribution as prediction
* custom confidence intervals as prediction
* robust data pre-processing module
* scikit-learn interface both for pre-processing and modeling
 

## Installing
This Python package can be installed using pip:

`pip install git+ssh://git@git.informatik.uni-leipzig.de/SWS/bayesify`

## Getting started
For now, you can refer to [the unit tests](tests/unit/pairwisetest.py) to see how to use bayesify. 