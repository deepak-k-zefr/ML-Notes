

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


## KNN vs k-means clustering

More reading : [How is the k-nearest neighbor algorithm different from k-means clustering?](http://stats.stackexchange.com/questions/56500/what-are-the-main-differences-between-k-means-and-k-nearest-neighbours)

K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm. While the mechanisms may seem similar at first, what this really means is that in order for K-Nearest Neighbors to work, you need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.

The critical difference here is that KNN needs labeled points and is thus supervised learning, while k-means doesn’t — and is thus unsupervised learning.

## KNN

the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:

In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

In k-NN regression, the output is the property value for the object. This value is the average of the values of its k nearest neighbors.


### Dimension reduction
For high-dimensional data (e.g., with number of dimensions more than 10) dimension reduction is usually performed prior to applying the k-NN algorithm in order to avoid the effects of the curse of dimensionality. 

The curse of dimensionality in the k-NN context basically means that Euclidean distance is unhelpful in high dimensions because all vectors are almost equidistant to the search query vector (imagine multiple points lying more or less on a circle with the query point at the center; the distance from the query to all data points in the search space is almost the same).

Feature extraction and dimension reduction can be combined in one step using principal component analysis (PCA), linear discriminant analysis (LDA), or canonical correlation analysis (CCA) techniques as a pre-processing step, followed by clustering by k-NN on feature vectors in reduced-dimension space. In machine learning this process is also called low-dimensional embedding



## Generalized linear model
A generalized linear model (GLM) is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

Intuition
Ordinary linear regression predicts the expected value of a given unknown quantity (the response variable, a random variable) as a linear combination of a set of observed values (predictors). This implies that a constant change in a predictor leads to a constant change in the response variable (i.e. a linear-response model). This is appropriate when the response variable has a normal distribution (intuitively, when a response variable can vary essentially indefinitely in either direction with no fixed "zero value", or more generally for any quantity that only varies by a relatively small amount, e.g. human heights).

However, these assumptions are inappropriate for some types of response variables. For example, in cases where the response variable is expected to be always positive and varying over a wide range, constant input changes lead to geometrically varying, rather than constantly varying, output changes. As an example, a prediction model might predict that 10 degree temperature decrease would lead to 1,000 fewer people visiting the beach is unlikely to generalize well over both small beaches (e.g. those where the expected attendance was 50 at a particular temperature) and large beaches (e.g. those where the expected attendance was 10,000 at a low temperature). The problem with this kind of prediction model would imply a temperature drop of 10 degrees would lead to 1,000 fewer people visiting the beach, a beach whose expected attendance was 50 at a higher temperature would now be predicted to have the impossible attendance value of −950.

Examples
When the response data, Y, are binary (taking on only values 0 and 1), the distribution function is generally chosen to be the Bernoulli distribution and the interpretation of μi is then the probability, p, of Yi taking on the value one.

There are several popular link functions for binomial functions.

Logit link function
The most typical link function is the canonical logit link:

![formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/8fafc094e76c824824b0b49467a84884525dad8e)
