# ML-Notes

SOURCES:

* Quora
* Wikipedia
* Cross Validates
* Springboard

[Supervised Learning](./docs/Supervised_learning.md)

[Performance Metrics](./docs/Performance_metrics.md)

[Bayesian Statistics](./docs/Bayesian_statistics.md)

[Probabilistic Graphical Models](./docs/Probabilistic_graphical_model.md)

[Good Answers from forums](./docs/Answers.md)




## Discriminative and Generative models

* [Generative vs. discriminative Stackoverflow](https://stats.stackexchange.com/questions/12421/generative-vs-discriminative)
* [Andrew Ng Generative Learning Algorithms](https://www.youtube.com/watch?v=z5UQyCESW64)
* [Generative vs Discriminative Good explanation](https://www.youtube.com/watch?v=OWJ8xVGRyFA)


## Remedies for multicollinearity

* Make sure you have not fallen into the dummy variable trap; including a dummy variable for every category (e.g., summer, autumn, winter, and spring) and including a constant term in the regression together guarantee perfect multicollinearity.

* Try seeing what happens if you use independent subsets of your data for estimation and apply those estimates to the whole data set. Theoretically you should obtain somewhat higher variance from the smaller datasets used for estimation, but the expectation of the coefficient values should be the same. Naturally, the observed coefficient values will vary, but look at how much they vary.

* Leave the model as is, despite multicollinearity. The presence of multicollinearity doesn't affect the efficacy of extrapolating the fitted model to new data provided that the predictor variables follow the same pattern of multicollinearity in the new data as in the data on which the regression model is based.

* Drop one of the variables. An explanatory variable may be dropped to produce a model with significant coefficients. However, you lose information (because you've dropped a variable). Omission of a relevant variable results in biased coefficient estimates for the remaining explanatory variables that are correlated with the dropped variable.

* Obtain more data, if possible. This is the preferred solution. More data can produce more precise parameter estimates (with lower standard errors), as seen from the formula in variance inflation factor for the variance of the estimate of a regression coefficient in terms of the sample size and the degree of multicollinearity.

* Mean-center the predictor variables.

* Standardize your independent variables. This may help reduce a false flagging of a condition index above 30.It has also been suggested that using the Shapley value, a game theory tool, the model could account for the effects of multicollinearity. The Shapley value assigns a value for each predictor and assesses all possible combinations of importance.

* Ridge regression or principal component regression or partial least squares regression can be used.

* If the correlated explanators are different lagged values of the same underlying explanator, then a distributed lag technique can be used, imposing a general structure on the relative values of the coefficients to be estimated.



