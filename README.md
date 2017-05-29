# ML-Notes

SOURCES:

* Quora
* Wikipedia
* Cross Validates
* Springboard

[Supervised Learning](./docs/Supervised_learning.md)

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



### Maximum likelihood estimation
Maximum likelihood estimation (MLE) is a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of making the observations given the parameters
 
For example, one may be interested in the heights of adult female penguins, but be unable to measure the height of every single penguin in a population due to cost or time constraints. Assuming that the heights are normally distributed with some unknown mean and variance, the mean and variance can be estimated with MLE while only knowing the heights of some sample of the overall population. MLE would accomplish this by taking the mean and variance as parameters and finding particular parametric values that make the observed results the most probable given the model.

X1, X2, X3, . . . Xn have joint density denoted as

 fθ(x1, x2, . . . , xn) = f(x1, x2, . . . , xn|θ)

Given observed values X1 = x1, X2 = x2, . . . , Xn = xn, the likelihood of θ is the function

lik(θ) = f(x1, x2, . . . , xn|θ)

If the distribution is discrete, f will be the frequency distribution function.
In words: lik(θ)=probability of observing the given data as a function of θ

The maximum likelihood estimate (mle) of θ is that value of θ that maximises lik(θ): it is
the value that makes the observed data the “most probable”.

Rather than maximising this product which can be quite tedious, we often use the fact
that the logarithm is an increasing function so it will be equivalent to maximise the log
likelihood

Discrete distribution, finite parameter space[edit]
Suppose one wishes to determine just how biased an unfair coin is. Call the probability of tossing a HEAD p. The goal then becomes to determine p.

Example:
Suppose the coin is tossed 80 times: i.e., the sample might be something like x1 = H, x2 = T, …, x80 = T, and the count of the number of HEADS "H" is observed.

The probability of tossing TAILS is 1 − p (so here p is θ above). Suppose the outcome is 49 HEADS and 31 TAILS, and suppose the coin was taken from a box containing three coins: one which gives HEADS with probability p = 1/3, one which gives HEADS with probability p = 1/2 and another which gives HEADS with probability p = 2/3. The coins have lost their labels, so which one it was is unknown. Using maximum likelihood estimation the coin that has the largest likelihood can be found, given the data that were observed. By using the probability mass function of the binomial distribution with sample size equal to 80, number successes equal to 49 but different values of p (the "probability of success"), the likelihood function (defined below) takes one of three values:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/36bc1e5127816685c557ccd68d4f4081d0b7f9fa)



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



## Probabilistic Graphical Models
# Markov Process
is a stochastic process that satisfies the Markov property. A process satisfies the Markov property if one can make predictions for the future of the process based solely on its present state just as well as one could knowing the process's full history, hence independently from such history; i.e., conditional on the present state of the system, its future and past states are independent.

A state diagram for a simple example is shown in the figure on the right, using a directed graph to picture the state transitions. The states represent whether a hypothetical stock market is exhibiting a bull market, bear market, or stagnant market trend during a given week. According to the figure, a bull week is followed by another bull week 90% of the time, a bear week 7.5% of the time, and a stagnant week the other 2.5% of the time. Labelling the state space {1 = bull, 2 = bear, 3 = stagnant} 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Finance_Markov_chain_example_state_space.svg/800px-Finance_Markov_chain_example_state_space.svg.png)

the transition matrix for this example is:
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/6cea2dc36a546e141ce2d072636dbf8a0005f235)


![](https://wikimedia.org/api/rest_v1/media/math/render/svg/df26d8a65d9997bd816356f0ebc532c46ea9a46c)

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


