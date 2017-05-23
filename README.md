# ML-questions-and-answers

SOURCES:

* Quora
* Springboard
* Wikipedia
* Cross Validates

# Assumptions of Various Models

## Linear Regression

* Linear relationship
* Multivariate normality
* No or little multicollinearity
* No auto-correlation(Autocorrelation occurs when the residuals are not independent from each other.  In other words when the value of y(x+1) is not independent from the value of y(x).)
* Homoscedasticity

More reading : http://r-statistics.co/Assumptions-of-Linear-Regression.html.   
More reading : http://www.statisticssolutions.com/assumptions-of-linear-regression/.  
More reading : http://www.nitiphong.com/paper_pdf/OLS_Assumptions.pdf.  


## Logistic Regression
* No outliers.(Use z-scores, histograms, and k-means clustering, to identify and remove outliers
and analyze residuals to identify outliers in the regression)
* Independent errors.(Like OLS, error terms are assumed uncorrelated.)
* No multicollinearity.(Check zero-order correlation matrix for high values (ie r>0.7)

More Reading :https://onlinecourses.science.psu.edu/stat504/node/164

 Because logistic regression uses MLE rather than OLS, it avoids many
of the typical assumptions tested in statistical analysis.
* Does not assume normality of variables (both DV and IVs).
* Does not assume linearity between DV and IVs.
* Does not assume homoscedasticity.
* Does not assume normal errors.
* MLE allows more flexibility in the data and analysis because it has
fewer restrictions.

## Neural Networks
There is no  assumption on data, errors or targets. In theory a Neural Network can approximate any function and this is done without assumptions, it only depends on data and network configuration.

##  What’s the trade-off between bias and variance?

More reading : [Bias-Variance Tradeoff (Wikipedia)](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)

Bias is error due to erroneous or overly simplistic assumptions in the learning algorithm you’re using. This can lead to the model underfitting your data, making it hard for it to have high predictive accuracy and for you to generalize your knowledge from the training set to the test set.

Variance is error due to too much complexity in the learning algorithm you’re using. This leads to the algorithm being highly sensitive to high degrees of variation in your training data, which can lead your model to overfit the data. You’ll be carrying too much noise from your training data for your model to be very useful for your test data.

The bias-variance decomposition essentially decomposes the learning error from any algorithm by adding the bias, the variance and a bit of irreducible error due to noise in the underlying dataset. Essentially, if you make the model more complex and add more variables, you’ll lose bias but gain some variance — in order to get the optimally reduced amount of error, you’ll have to tradeoff bias and variance. You don’t want either high bias or high variance in your model.



##  How is KNN different from k-means clustering?

More reading : [How is the k-nearest neighbor algorithm different from k-means clustering?](http://stats.stackexchange.com/questions/56500/what-are-the-main-differences-between-k-means-and-k-nearest-neighbours)

K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm. While the mechanisms may seem similar at first, what this really means is that in order for K-Nearest Neighbors to work, you need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.

The critical difference here is that KNN needs labeled points and is thus supervised learning, while k-means doesn’t — and is thus unsupervised learning.



##  Explain how a ROC curve works.

More reading: [Receiver operating characteristic (Wikipedia)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).

### AUC
The AUC of a classifier is equal to the probability that the classifier will rank a randomly chosen positive example higher than a randomly chosen negative example, i.e. P(score(x+)>score(x−))

### Sensitivity And Specificity

Sensitivity refers to the test's ability to correctly detect patients who do have the condition.In the example of a medical test used to identify a disease, the sensitivity of the test is the proportion of people who test positive for the disease among those who have the disease. Mathematically, this can be expressed as:

![Sensitivity](https://wikimedia.org/api/rest_v1/media/math/render/svg/fbad73213a4578685fefa43ec96ce53533057e11)

Specificity relates to the test's ability to correctly detect patients without a condition.  Consider the example of a medical test for diagnosing a disease. Specificity of a test is the proportion of healthy patients known not to have the disease, who will test negative for it. Mathematically, this can also be written as:


![Specificity](https://wikimedia.org/api/rest_v1/media/math/render/svg/d7856a809dafad4fa9566eef65b37bedeaa53132)

[Image of High Sensitivity Low Specificity](https://en.wikipedia.org/wiki/File:HighSensitivity_LowSpecificity_1401x1050.png)  Lesser False Negatives and more False Positives. Detect more people with the disease.

[Image of Low Sensitivity High Specificity](https://en.wikipedia.org/wiki/File:LowSensitivity_HighSpecificity_1400x1050.png)
Lesser False Positives and more False Negatives. Detect more people without the disease.

[Worked out example image](https://en.wikipedia.org/wiki/Template:SensSpecPPVNPV)

##  Define precision and recall.
More Reading: [precision and recall-Scikit Learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

Precision (P) is defined as the number of true positives (T_p) over the number of true positives plus the number of false positives (F_p).

P = T_p/(T_p+F_p)

Recall (R) is defined as the number of true positives (T_{p}) over the number of true positives plus the number of false negatives (F_n).

R = (T_p)/(T_p+F_n)

These quantities are also related to the (F_1) score, which is defined as the harmonic mean of precision and recall.

## What is Bayes’ Theorem? How is it useful in a machine learning context?

[More reading: An Intuitive (and Short) Explanation of Bayes’ Theorem (BetterExplained)](https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/)

Bayes’ Theorem gives you the posterior probability of an event given what is known as prior knowledge.

Mathematically, it’s expressed as the true positive rate of a condition sample divided by the sum of the false positive rate of the population and the true positive rate of a condition. Say you had a 60% chance of actually having the flu after a flu test, but out of people who had the flu, the test will be false 50% of the time, and the overall population only has a 5% chance of having the flu. Would you actually have a 60% chance of having the flu after having a positive test?

Bayes’ Theorem says no. It says that you have a (.6 * 0.05) (True Positive Rate of a Condition Sample) / (.6*0.05)(True Positive Rate of a Condition Sample) + (.5*0.95) (False Positive Rate of a Population)  = 0.0594 or 5.94% chance of getting a flu.

Bayes’ Theorem is the basis behind a branch of machine learning that most notably includes the Naive Bayes classifier. That’s something important to consider when you’re faced with machine learning interview questions.

## What is the difference between Discriminative an Generative models?
* [Generative vs. discriminative Stackoverflow](https://stats.stackexchange.com/questions/12421/generative-vs-discriminative)
* [Andrew Ng Generative Learning Algorithms](https://www.youtube.com/watch?v=z5UQyCESW64)
* [Generative vs Discriminative Good explanation](https://www.youtube.com/watch?v=OWJ8xVGRyFA)


## Remedies for multicollinearity
* Make sure you have not fallen into the dummy variable trap; including a dummy variable for every category (e.g., summer, autumn, winter, and spring) and including a constant term in the regression together guarantee perfect multicollinearity.
* Try seeing what happens if you use independent subsets of your data for estimation and apply those estimates to the whole data set. Theoretically you should obtain somewhat higher variance from the smaller datasets used for estimation, but the expectation of the coefficient values should be the same. Naturally, the observed coefficient values will vary, but look at how much they vary.
* Leave the model as is, despite multicollinearity. The presence of multicollinearity doesn't affect the efficacy of extrapolating the fitted model to new data provided that the predictor variables follow the same pattern of multicollinearity in the new data as in the data on which the regression model is based.[9]
* Drop one of the variables. An explanatory variable may be dropped to produce a model with significant coefficients. However, you lose information (because you've dropped a variable). Omission of a relevant variable results in biased coefficient estimates for the remaining explanatory variables that are correlated with the dropped variable.
* Obtain more data, if possible. This is the preferred solution. More data can produce more precise parameter estimates (with lower standard errors), as seen from the formula in variance inflation factor for the variance of the estimate of a regression coefficient in terms of the sample size and the degree of multicollinearity.
* Mean-center the predictor variables. Generating polynomial terms (i.e., for {\displaystyle x_{1}} x_{1}, {\displaystyle x_{1}^{2}} x_{1}^{2}, {\displaystyle x_{1}^{3}} x_{1}^{3}, etc.) or interaction terms (i.e., {\displaystyle x_{1}\times x_{2}} {\displaystyle x_{1}\times x_{2}}, etc.) can cause some multicollinearity if the variable in question has a limited range (e.g., [2,4]). Mean-centering will eliminate this special kind of multicollinearity. However, in general, this has no effect. It can be useful in overcoming problems arising from rounding and other computational steps if a carefully designed computer program is not used.
* Standardize your independent variables. This may help reduce a false flagging of a condition index above 30.
It has also been suggested that using the Shapley value, a game theory tool, the model could account for the effects of multicollinearity. The Shapley value assigns a value for each predictor and assesses all possible combinations of importance.[10]
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


