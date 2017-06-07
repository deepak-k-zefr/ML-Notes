## Multicollinearity

Multicollinearity (also collinearity) is a phenomenon in which two or more predictor variables in a multiple regression model are highly correlated, meaning that one can be linearly predicted from the others with a substantial degree of accuracy. In this situation the coefficient estimates of the multiple regression may change erratically in response to small changes in the  model or the data.

 Multicollinearity does not reduce the predictive power or reliability of the model as a whole, at least within the sample data set; it only affects calculations regarding individual predictors. 
 
 That is, a multiple regression model with correlated predictors can indicate how well the entire bundle of predictors predicts the outcome variable, but it may not give valid results about any individual predictor, or about which predictors are redundant with respect to others.

### Consequences of multicollinearity

In the presence of multicollinearity, the estimate of one variable's impact on the dependent variable Y while controlling for the others tends to be less precise than if predictors were uncorrelated with one another. The usual interpretation of a regression coefficient is that it provides an estimate of the effect of a one unit change in an independent variable, X_1 holding the other variables constant. If X_1 is highly correlated with another independent variable,  X_2, in the given data set, then we have a set of observations for which X_1 and X_2 have a particular linear stochastic relationship. We don't have a set of observations for which all changes in X_1 are independent of changes in X_2, so we have an imprecise estimate of the effect of independent changes in  X_1.

In some sense, the collinear variables contain the same information about the dependent variable. If nominally "different" measures actually quantify the same phenomenon then they are redundant. Alternatively, if the variables are accorded different names and perhaps employ different numeric measurement scales but are highly correlated with each other, then they suffer from redundancy.

One of the features of multicollinearity is that the standard errors of the affected coefficients tend to be large. In that case, the test of the hypothesis that the coefficient is equal to zero may lead to a failure to reject a false null hypothesis of no effect of the explanator, a type II error.

Another issue with multicollinearity is that small changes to the input data can lead to large changes in the model, even resulting in changes of sign of parameter estimates.[2]

A principal danger of such data redundancy is that of overfitting in regression analysis models. The best regression models are those in which the predictor variables each correlate highly with the dependent (outcome) variable but correlate at most only minimally with each other. Such a model is often called "low noise" and will be statistically robust (that is, it will predict reliably across numerous samples of variable sets drawn from the same statistical population).

So long as the underlying specification is correct, multicollinearity does not actually bias results; it just produces large standard errors in the related independent variables. More importantly, the usual use of regression is to take coefficients from the model and then apply them to other data. Since multicollinearity causes imprecise estimates of coefficient values, the resulting out-of-sample predictions will also be imprecise. And if the pattern of multicollinearity in the new data differs from that in the data that was fitted, such extrapolation may introduce large errors in the predictions


### Remedies for multicollinearity

* Make sure you have not fallen into the dummy variable trap; including a dummy variable for every category (e.g., summer, autumn, winter, and spring) and including a constant term in the regression together guarantee perfect multicollinearity.

* Try seeing what happens if you use independent subsets of your data for estimation and apply those estimates to the whole data set. Theoretically you should obtain somewhat higher variance from the smaller datasets used for estimation, but the expectation of the coefficient values should be the same. Naturally, the observed coefficient values will vary, but look at how much they vary.

* Leave the model as is, despite multicollinearity. The presence of multicollinearity doesn't affect the efficacy of extrapolating the fitted model to new data provided that the predictor variables follow the same pattern of multicollinearity in the new data as in the data on which the regression model is based.

* Drop one of the variables. An explanatory variable may be dropped to produce a model with significant coefficients. However, you lose information (because you've dropped a variable). Omission of a relevant variable results in biased coefficient estimates for the remaining explanatory variables that are correlated with the dropped variable.

* Obtain more data, if possible. This is the preferred solution. More data can produce more precise parameter estimates (with lower standard errors), as seen from the formula in variance inflation factor for the variance of the estimate of a regression coefficient in terms of the sample size and the degree of multicollinearity.

* Mean-center the predictor variables.

* Standardize your independent variables. This may help reduce a false flagging of a condition index above 30.It has also been suggested that using the Shapley value, a game theory tool, the model could account for the effects of multicollinearity. The Shapley value assigns a value for each predictor and assesses all possible combinations of importance.

* Ridge regression or principal component regression or partial least squares regression can be used.

* If the correlated explanators are different lagged values of the same underlying explanator, then a distributed lag technique can be used, imposing a general structure on the relative values of the coefficients to be estimated.
