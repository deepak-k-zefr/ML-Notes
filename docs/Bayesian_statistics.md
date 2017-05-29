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



