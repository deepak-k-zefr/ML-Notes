# NLP



### tf–idf
tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

The tf-idf value increases proportionally to the
* number of times a word appears in the document, 
* but is often offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/10109d0e60cc9d50a1ea2f189bac0ac29a030a00)

Term frequency
The weight of a term that occurs in a document is simply proportional to the term frequency

Inverse document frequency
The specificity of a term can be quantified as an inverse function of the number of documents in which the word occurs


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

