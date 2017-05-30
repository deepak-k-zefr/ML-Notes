

## Linear Regression

* **Linear relationship (Model is linear in parameters)**
The linearity assumption can best be observed in scatter plots

* **No or little multicollinearity**
There is no perfect linear relationship between explanatory variables. Multicollinearity occurs when the independent variables are not independent from each other. 
  * Correlation matrix – when computing the matrix of Pearson's Bivariate Correlation among all independent variables the correlation coefficients need to be smaller than 1.
  * VIF is a metric computed for every X variable that goes into a linear model. If the VIF of a variable is high, it means the information in that variable is already explained by other X variables present in the given model, which means, more redundant is that variable. So, lower the VIF (<2) the better

* **Normality of residuals**

* **The mean of residuals is zero**

* **No auto-correlation**
(Autocorrelation occurs when the residuals are not independent from each other.  In other words when the value of y(x+1) is not independent from the value of y(x).) It is observed in time series data.

* **Homoscedasticity of residuals or equal variance**

More reading : http://r-statistics.co/Assumptions-of-Linear-Regression.html.   
More reading : http://www.statisticssolutions.com/assumptions-of-linear-regression/.  
More reading : http://www.nitiphong.com/paper_pdf/OLS_Assumptions.pdf.  

## Gauss–Markov theorem

In statistics, the Gauss–Markov theorem, states that in a linear regression model in which the errors have expectation zero and are uncorrelated and have equal variances, the best linear unbiased estimator (BLUE) of the coefficients is given by the ordinary least squares (OLS) estimator, provided it exists. 

Here "best" means giving the lowest variance of the estimate, as compared to other unbiased, linear estimators. The errors do not need to be normal, nor do they need to be independent and identically distributed (only uncorrelated with mean zero and homoscedastic with finite variance). The requirement that the estimator be unbiased cannot be dropped, since biased estimators exist with lower variance.


## Logistic Regression

* **No outliers.(Use z-scores, histograms, and k-means clustering, to identify and remove outliers**
and analyze residuals to identify outliers in the regression)
* **Independent errors.(Like OLS, error terms are assumed uncorrelated.)**
* **No multicollinearity.(Check zero-order correlation matrix for high values (ie r>0.7)**

Logistic regression is considered a generalized linear model because the outcome always depends on the sum of the inputs and parameters. Or in other words, the output cannot depend on the product (or quotient, etc.) of its parameters! There's no interaction between the parameter weights, nothing like w_1*x_1 * w_2* x_2 or so, which would make our model non-linear!

### Definition of the logistic function

An explanation of logistic regression can begin with an explanation of the standard logistic function. The logistic function is useful because it can take any real input  *t*, whereas the output always takes values between zero and one and hence is interpretable as a probability. 

![formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/5e648e1dd38ef843d57777cd34c67465bbca694f)


The logistic function sigma (t) is defined as follows:

![fig](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png)

![fig](https://wikimedia.org/api/rest_v1/media/math/render/svg/836d93163447344be4715ec00638c1cd829e376c)

![fig](https://wikimedia.org/api/rest_v1/media/math/render/svg/57fa62921bfe1721bca86f8db39f44f4c1094cd5)


More Reading :https://onlinecourses.science.psu.edu/stat504/node/164

 Because logistic regression uses MLE rather than OLS, it avoids many
of the typical assumptions tested in statistical analysis.
* Does not assume normality of variables (both DV and IVs).
* Does not assume linearity between DV and IVs.
* Does not assume homoscedasticity.
* Does not assume normal errors.
* MLE allows more flexibility in the data and analysis because it has fewer restrictions.


## Generalized linear model
GLM is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution

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

## Neural Networks

There is no  assumption on data, errors or targets. In theory a Neural Network can approximate any function and this is done without assumptions, it only depends on data and network configuration.


## Ensemble methods

The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.

Two families of ensemble methods are usually distinguished:

In averaging methods, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.

Examples: Bagging methods, Forests of randomized trees, ...

By contrast, in boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.

## Random forest

### Motivation

Trees that are grown very deep tend to learn highly irregular patterns: they overfit their training sets, i.e. have low bias, but very high variance. Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set, with the goal of reducing the variance.[3]:587–588 This comes at the expense of a small increase in the bias and some loss of interpretability, but generally greatly boosts the performance in the final model.


### Tree bagging

The training algorithm for random forests applies the general technique of bootstrap aggregating, or bagging, to tree learners. Given a training set X = x1, ..., xn with responses Y = y1, ..., yn, bagging repeatedly (B times) selects a random sample with replacement of the training set and fits trees to these samples:

For b = 1, ..., B:

   1) Sample, with replacement, B training examples from X, Y; call these Xb, Yb.
   2) Train a decision or regression tree fb on Xb, Yb.
   After training, predictions for unseen samples x' can be made by averaging the predictions from all the individual       regression trees on x':

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/b54befce12aefdb29442bfc71cb5ad452364e8d8)

or by taking the majority vote in the case of decision trees.

This bootstrapping procedure leads to better model performance because it decreases the variance of the model, without increasing the bias. This means that while the predictions of a single tree are highly sensitive to noise in its training set, the average of many trees is not, as long as the trees are not correlated. Simply training many trees on a single training set would give strongly correlated trees (or even the same tree many times, if the training algorithm is deterministic); bootstrap sampling is a way of de-correlating the trees by showing them different training sets.

The number of samples/trees, B, is a free parameter. Typically, a few hundred to several thousand trees are used, depending on the size and nature of the training set. An optimal number of trees B can be found using cross-validation, or by observing the out-of-bag error: the mean prediction error on each training sample xᵢ, using only the trees that did not have xᵢ in their bootstrap sample. The training and test error tend to level off after some number of trees have been fit.


### From bagging to random forests

The above procedure describes the original bagging algorithm for trees. Random forests differ in only one way from this general scheme: they use a modified tree learning algorithm that selects, at each candidate split in the learning process, a random subset of the features. This process is sometimes called "feature bagging". The reason for doing this is the correlation of the trees in an ordinary bootstrap sample: if one or a few features are very strong predictors for the response variable (target output), these features will be selected in many of the B trees, causing them to become correlated. An analysis of how bagging and random subspace projection contribute to accuracy gains under different conditions is given by Ho.

Typically, for a classification problem with p features, √p (rounded down) features are used in each split. For regression problems the inventors recommend p/3 (rounded down) with a minimum node size of 5 as the default.

![](https://databricks.com/wp-content/uploads/2015/01/Ensemble-example.png)

## Bias Variance Trade-off

More reading : [Bias-Variance Tradeoff (Wikipedia)](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)

Bias is error due to erroneous or overly simplistic assumptions in the learning algorithm you’re using. This can lead to the model underfitting your data, making it hard for it to have high predictive accuracy and for you to generalize your knowledge from the training set to the test set.

Variance is error due to too much complexity in the learning algorithm you’re using. This leads to the algorithm being highly sensitive to high degrees of variation in your training data, which can lead your model to overfit the data. You’ll be carrying too much noise from your training data for your model to be very useful for your test data.

The bias-variance decomposition essentially decomposes the learning error from any algorithm by adding the bias, the variance and a bit of irreducible error due to noise in the underlying dataset. Essentially, if you make the model more complex and add more variables, you’ll lose bias but gain some variance — in order to get the optimally reduced amount of error, you’ll have to tradeoff bias and variance. You don’t want either high bias or high variance in your model.

![Graphical illustration of bias and varianceFrom Understanding the Bias-Variance Tradeoff, by Scott Fortmann-Roe.](http://www.kdnuggets.com/wp-content/uploads/bias-and-variance.jpg)


## KNN vs k-means clustering

More reading : [How is the k-nearest neighbor algorithm different from k-means clustering?](http://stats.stackexchange.com/questions/56500/what-are-the-main-differences-between-k-means-and-k-nearest-neighbours)

K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm. While the mechanisms may seem similar at first, what this really means is that in order for K-Nearest Neighbors to work, you need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.

The critical difference here is that KNN needs labeled points and is thus supervised learning, while k-means doesn’t — and is thus unsupervised learning.




### Naive Bayes Text	Classificaion

Input:	
• 	a	document	d	
• a	fixed	set	of	classes		C	= {c1,	c2,…,	cJ}	
• A	training	set	of	m hand-labeled	documents	(d1,c1),....,(dm,cm)	

Output:	
a	predicted	class	c ∈ C


Simple	(“naïve”)	classifica1on	method	based	on	
Bayes	rule	
• Relies	on	very	simple	representa1on	of	document	
• Bag	of	words	

•For	a	document	d and	a	class	c

P(d/c)=P(c/d)*P(d)/ P(c)


Naïve	Bayes	Classifier	(I)	

MAP is “maximum a posteriori” = most likely class 


Cmap= argmax P(c|d) . (c belongs to C)
     = P(d/c)*P(c)


https://web.stanford.edu/class/cs124/lec/naivebayes.pdf





