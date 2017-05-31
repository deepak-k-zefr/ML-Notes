## Algorithms for hyperparameter optimization



### Grid search

The traditional way of performing hyperparameter optimization has been grid search, or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set[2] or evaluation on a held-out validation set.

Since the parameter space of a machine learner may include real-valued or unbounded value spaces for certain parameters, manually set bounds and discretization may be necessary before applying grid search.

For example, a typical soft-margin SVM classifier equipped with an RBF kernel has at least two hyperparameters that need to be tuned for good performance on unseen data: a regularization constant C and a kernel hyperparameter γ. Both parameters are continuous, so to perform grid search, one selects a finite set of "reasonable" values for each, say

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/4124e15320f26a727f12f02d9bc61edc512878fd)
Grid search then trains an SVM with each pair (C, γ) in the Cartesian product of these two sets and evaluates their performance on a held-out validation set (or by internal cross-validation on the training set, in which case multiple SVMs are trained per pair). Finally, the grid search algorithm outputs the settings that achieved the highest score in the validation procedure.

Grid search suffers from the curse of dimensionality, but is often embarrassingly parallel because typically the hyperparameter settings it evaluates are independent of each other


### Random search

Since grid searching is an exhaustive and therefore potentially expensive method, several alternatives have been proposed. In particular, a randomized search that simply samples parameter settings a fixed number of times has been found to be more effective in high-dimensional spaces than exhaustive search. This is because oftentimes, it turns out some hyperparameters do not significantly affect the loss. Therefore, having randomly dispersed data gives more "textured" data than an exhaustive search over parameters that ultimately do not affect the loss.



## Curse of dimensionality

Let's say you have a straight line 100 yards long and you dropped a penny somewhere on it. It wouldn't be too hard to find. You walk along the line and it takes two minutes.

Now let's say you have a square 100 yards on each side and you dropped a penny somewhere on it. It would be pretty hard, like searching across two football fields stuck together. It could take days.

Now a cube 100 yards across. That's like searching a 30-story building the size of a football stadium. Ugh.

The difficulty of searching through the space gets a *lot* harder as you have more dimensions. You might not realize this intuitively when it's just stated in mathematical formulas, since they all have the same "width". That's the curse of dimensionality. It gets to have a name because it is unintuitive, useful, and yet simple.


## Bias Variance Trade-off

More reading : [Bias-Variance Tradeoff (Wikipedia)](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)

Bias is error due to erroneous or overly simplistic assumptions in the learning algorithm you’re using. This can lead to the model underfitting your data, making it hard for it to have high predictive accuracy and for you to generalize your knowledge from the training set to the test set.

Variance is error due to too much complexity in the learning algorithm you’re using. This leads to the algorithm being highly sensitive to high degrees of variation in your training data, which can lead your model to overfit the data. You’ll be carrying too much noise from your training data for your model to be very useful for your test data.

The bias-variance decomposition essentially decomposes the learning error from any algorithm by adding the bias, the variance and a bit of irreducible error due to noise in the underlying dataset. Essentially, if you make the model more complex and add more variables, you’ll lose bias but gain some variance — in order to get the optimally reduced amount of error, you’ll have to tradeoff bias and variance. You don’t want either high bias or high variance in your model.

![Graphical illustration of bias and varianceFrom Understanding the Bias-Variance Tradeoff, by Scott Fortmann-Roe.](http://www.kdnuggets.com/wp-content/uploads/bias-and-variance.jpg)


## Lasso

lasso (least absolute shrinkage and selection operator) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produce

It penalizes the absolute size of the regression coefficients

![Lasso eqn](https://wikimedia.org/api/rest_v1/media/math/render/svg/2904b78ec712617fdef0bd35e28442b9c1b35b03)

Here  *t* is a prespecified free parameter that determines the amount of regularisation

![Lasso vs Ridge](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/L1_and_L2_balls.svg/1600px-L1_and_L2_balls.svg.png)


## Ridge Regression

Motivation: too many predictors

It is not unusual to see the number of input variables greatly exceed the number of observations, e.g. micro-array data analysis, environmental pollution studies.

With many predictors, fitting the full model without penalization will result in large prediction intervals, and LS regression estimator may not uniquely exist.


## Elastic Net

 Elastic net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods
 The elastic net method overcomes the limitations of the LASSO (least absolute shrinkage and selection operator) method which uses a penalty function based on

![Lasso Penalty](https://wikimedia.org/api/rest_v1/media/math/render/svg/5a188f4b162086fb06a4485f3336baefc22e18b3)

Use of this penalty function has several limitations.[1] For example, in the "large p, small n" case (high-dimensional data with few examples), the LASSO selects at most n variables before it saturates. Also if there is a group of highly correlated variables, then the LASSO tends to select one variable from a group and ignore the others. To overcome these limitations, the elastic net adds a quadratic part to the penalty, which when used alone is ridge regression (known also as Tikhonov regularization). The estimates from the elastic net method are defined by


![](https://wikimedia.org/api/rest_v1/media/math/render/svg/48b3ad7bcf1954b906d16dde0c1d3b65ca8d45aa)
