# ML-Notes

SOURCES:

* Quora
* Wikipedia
* Cross Validates
* Springboard


## Linear Regression

* Linear relationship (Model is linear in parameters)
The linearity assumption can best be tested with scatter plots

* No or little multicollinearity
There is no perfect linear relationship between explanatory variables. Multicollinearity occurs when the independent variables are not independent from each other. 
  * Correlation matrix – when computing the matrix of Pearson's Bivariate Correlation among all independent variables the correlation coefficients need to be smaller than 1.
  * VIF is a metric computed for every X variable that goes into a linear model. If the VIF of a variable is high, it means the information in that variable is already explained by other X variables present in the given model, which means, more redundant is that variable. So, lower the VIF (<2) the better

* Normality of residuals

* The mean of residuals is zero

* No auto-correlation(Autocorrelation occurs when the residuals are not independent from each other.  In other words when the value of y(x+1) is not independent from the value of y(x).)

* Homoscedasticity of residuals or equal variance

More reading : http://r-statistics.co/Assumptions-of-Linear-Regression.html.   
More reading : http://www.statisticssolutions.com/assumptions-of-linear-regression/.  
More reading : http://www.nitiphong.com/paper_pdf/OLS_Assumptions.pdf.  

## Gauss–Markov theorem

In statistics, the Gauss–Markov theorem, states that in a linear regression model in which the errors have expectation zero and are uncorrelated and have equal variances, the best linear unbiased estimator (BLUE) of the coefficients is given by the ordinary least squares (OLS) estimator, provided it exists. 

Here "best" means giving the lowest variance of the estimate, as compared to other unbiased, linear estimators. The errors do not need to be normal, nor do they need to be independent and identically distributed (only uncorrelated with mean zero and homoscedastic with finite variance). The requirement that the estimator be unbiased cannot be dropped, since biased estimators exist with lower variance.


## Logistic Regression

* No outliers.(Use z-scores, histograms, and k-means clustering, to identify and remove outliers
and analyze residuals to identify outliers in the regression)
* Independent errors.(Like OLS, error terms are assumed uncorrelated.)
* No multicollinearity.(Check zero-order correlation matrix for high values (ie r>0.7)

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


##   ROC curve 

More reading: [Receiver operating characteristic (Wikipedia)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).



### AUC

The AUC of a classifier is equal to the probability that the classifier will rank a randomly chosen positive example higher than a randomly chosen negative example, i.e. P(score(x+)>score(x−))



### Sensitivity And Specificity

Sensitivity refers to the test's ability to correctly detect patients who do have the condition.In the example of a medical test used to identify a disease, the sensitivity of the test is the proportion of people who test positive for the disease among those who have the disease. Mathematically, this can be expressed as:

![Sensitivity](https://wikimedia.org/api/rest_v1/media/math/render/svg/fbad73213a4578685fefa43ec96ce53533057e11)

Specificity relates to the test's ability to correctly detect patients without a condition.  Consider the example of a medical test for diagnosing a disease. Specificity of a test is the proportion of healthy patients known not to have the disease, who will test negative for it. Mathematically, this can also be written as:


![Specificity](https://wikimedia.org/api/rest_v1/media/math/render/svg/d7856a809dafad4fa9566eef65b37bedeaa53132)

![Image of High Sensitivity Low Specificity](https://upload.wikimedia.org/wikipedia/commons/e/e2/HighSensitivity_LowSpecificity_1401x1050.png)

Lesser False Negatives and more False Positives. Detect more people with the disease.


[Worked out example image](https://en.wikipedia.org/wiki/Template:SensSpecPPVNPV)


## Precision and Recall.

More Reading: [precision and recall-Scikit Learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)


[Precision and Recall](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)!

Precision (P) is defined as the number of true positives (Tp) over the number of true positives plus the number of false positives (Fp).

![Precision =](https://wikimedia.org/api/rest_v1/media/math/render/svg/26106935459abe7c266f7b1ebfa2a824b334c807)

Recall (R) is defined as the number of true positives (Tp}) over the number of true positives plus the number of false negatives (Fn).

![Recall =](https://wikimedia.org/api/rest_v1/media/math/render/svg/4c233366865312bc99c832d1475e152c5074891b)

These quantities are also related to the (F1) score, which is defined as the harmonic mean of precision and recall.




##  Bayes’ Theorem

[More reading: An Intuitive (and Short) Explanation of Bayes’ Theorem (BetterExplained)](https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/)

Bayes’ Theorem gives you the posterior probability of an event given what is known as prior knowledge.

Mathematically, it’s expressed as the true positive rate of a condition sample divided by the sum of the false positive rate of the population and the true positive rate of a condition. Say you had a 60% chance of actually having the flu after a flu test, but out of people who had the flu, the test will be false 50% of the time, and the overall population only has a 5% chance of having the flu. Would you actually have a 60% chance of having the flu after having a positive test?

Bayes’ Theorem says no. It says that you have a (.6 * 0.05) (True Positive Rate of a Condition Sample) / (.6*0.05)(True Positive Rate of a Condition Sample) + (.5*0.95) (False Positive Rate of a Population)  = 0.0594 or 5.94% chance of getting a flu.

Bayes’ Theorem is the basis behind a branch of machine learning that most notably includes the Naive Bayes classifier. That’s something important to consider when you’re faced with machine learning interview questions.




## Naive Bayes spam filtering

Particular words have particular probabilities of occurring in spam email and in legitimate email. For instance, most email users will frequently encounter the word "Viagra" in spam email, but will seldom see it in other email. The filter doesn't know these probabilities in advance, and must first be trained so it can build them up. To train the filter, the user must manually indicate whether a new email is spam or not. For all words in each training email, the filter will adjust the probabilities that each word will appear in spam or legitimate email in its database. For instance, Bayesian spam filters will typically have learned a very high spam probability for the words "Viagra" and "refinance", but a very low spam probability for words seen only in legitimate email, such as the names of friends and family members

After training, the word probabilities (also known as likelihood functions) are used to compute the probability that an email with a particular set of words in it belongs to either category. Each word in the email contributes to the email's spam probability, or only the most interesting words. This contribution is called the posterior probability and is computed using Bayes' theorem. Then, the email's spam probability is computed over all words in the email, and if the total exceeds a certain threshold (say 95%), the filter will mark the email as a spam.

Bayesian email filters utilize Bayes' theorem. Bayes' theorem is used several times in the context of spam:

### Mathematical foundation

* a first time, to compute the probability that the message is spam, knowing that a given word appears in this message;
* a second time, to compute the probability that the message is spam, taking into consideration all of its words (or a relevant subset of them);
* sometimes a third time, to deal with rare words.



Let's suppose the suspected message contains the word "replica". Most people who are used to receiving e-mail know that this message is likely to be spam, more precisely a proposal to sell counterfeit copies of well-known brands of watches. The spam detection software, however, does not "know" such facts; all it can do is compute probabilities.

The formula used by the software to determine that, is derived from Bayes' theorem
![Bayes Theorem](https://wikimedia.org/api/rest_v1/media/math/render/svg/dc8c39ec48e65c0ab10dabe343d4da9a9585a77b)

where:

Pr(S|W) is the probability that a message is a spam, knowing that the word "replica" is in it;
Pr(S) is the overall probability that any given message is spam;
Pr(W|S) is the probability that the word "replica" appears in spam messages;
Pr(H) is the overall probability that any given message is not spam (is "ham");
Pr(W|H) is the probability that the word "replica" appears in ham messages.

Computing the probability that a message containing a given word is spam


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

## From Cross Validated


[What is the difference between “likelihood” and “probability”?](https://stats.stackexchange.com/questions/2641/what-is-the-difference-between-likelihood-and-probability).  
[Is there a standard and accepted method for selecting the number of layers, and the number of nodes in each layer, in a feed-forward neural network? I'm interested in automated ways of building neural networks.](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw).  
[What does the hidden layer in a neural network compute?](http://stats.stackexchange.com/questions/63152/what-does-the-hidden-layer-in-a-neural-network-compute?rq=1).  
[What does O(log n) mean exactly?](http://stackoverflow.com/questions/2307283/what-does-olog-n-mean-exactly?noredirect=1&lq=1).    
[Bayesian and frequentist reasoning in plain English](http://stats.stackexchange.com/questions/22/bayesian-and-frequentist-reasoning-in-plain-english).  
[How to choose the number of hidden layers and nodes in a feedforward neural network?](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw).  
[Explaining to laypeople why bootstrapping works](http://stats.stackexchange.com/questions/26088/explaining-to-laypeople-why-bootstrapping-works).  
[Can someone help to explain the difference between independent and random?](http://stats.stackexchange.com/questions/231425/can-someone-help-to-explain-the-difference-between-independent-and-random?noredirect=1&lq=1).  
[When should I use lasso vs ridge?](http://stats.stackexchange.com/questions/866/when-should-i-use-lasso-vs-ridge).  
[Relationship between SVD and PCA. How to use SVD to perform PCA?](http://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca).  
[How to reverse PCA and reconstruct original variables from several principal](http://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com?rq=1).  
[Bagging, boosting and stacking in machine learning](http://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning).    
[In linear regression, when is it appropriate to use the log of an independent variable instead of the actual values?](http://stats.stackexchange.com/questions/298/in-linear-regression-when-is-it-appropriate-to-use-the-log-of-an-independent-va).  
[How to interpret a QQ plot](http://stats.stackexchange.com/questions/101274/how-to-interpret-a-qq-plot).    



## Other Sources
[Complexity of Python Operations](https://www.ics.uci.edu/~pattis/ICS-33/lectures/complexitypython.txt).  
[Wiki - Complexity of Python Operations](https://wiki.python.org/moin/TimeComplexity).  
[Everything about R^2](http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit) . 


## Sebastian Raschka
[What-is-the-role-of-the-activation-function-in-a-neural-network](https://www.quora.com/What-is-the-role-of-the-activation-function-in-a-neural-network).     
[What's the difference between gradient descent and stochastic gradient descent?](https://www.quora.com/Whats-the-difference-between-gradient-descent-and-stochastic-gradient-descent/answer/Sebastian-Raschka-1?srid=9yUC).   
[How do I select SVM kernels?](https://www.quora.com/How-do-I-select-SVM-kernels/answer/Sebastian-Raschka-1?srid=9yUC).   
[What is the best visual explanation for the back propagation algorithm for neural networks?](https://www.quora.com/What-is-the-best-visual-explanation-for-the-back-propagation-algorithm-for-neural-networks/answer/Sebastian-Raschka-1?srid=9yUC).  
[How do I debug an artificial neural network algorithm?](https://www.quora.com/How-do-I-debug-an-artificial-neural-network-algorithm/answer/Sebastian-Raschka-1?srid=9yUC).  


## From Quora
[When should we use logistic regression and Neural Network?](https://www.quora.com/When-should-we-use-logistic-regression-and-Neural-Network/answer/Sebastian-Raschka-1?srid=9yUC).   
[What are Kernels in Machine Learning and SVM?](https://www.quora.com/What-are-Kernels-in-Machine-Learning-and-SVM)
[Supervised Learning Topic FAQ](https://www.quora.com/topic/Supervised-Learning/faq).   
[What are the advantages of logistic regression over decision trees?](https://www.quora.com/What-are-the-advantages-of-logistic-regression-over-decision-trees).   


