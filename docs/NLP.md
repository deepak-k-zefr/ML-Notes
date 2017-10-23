# NLP



### tf–idf
tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

The tf-idf value increases proportionally to the
* number of times a word appears in the document, 
* but is often offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/10109d0e60cc9d50a1ea2f189bac0ac29a030a00)

#### Term frequency
The weight of a term that occurs in a document is simply proportional to the term frequency
measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: 

TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).


#### Inverse document frequency
The specificity of a term can be quantified as an inverse function of the number of documents in which the word occurs

IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: 

IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

#### EXAMPLE
Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.




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




### Naive Bayes Text	Classificaion

Input:	
* a	document	`d`
* a	fixed	set	of	classes		`C	= {c1,	c2,…,	cJ}`	
* A	training	set	of	m hand-labeled	documents	`(d1,c1),....,(dm,cm)`	

Output:	
a	predicted	class	`c ∈ C`


Simple	(“naïve”)	classifica1on	method	based	on	
Bayes	rule	
* Relies	on	very	simple	representa1on	of	document	
* Bag	of	words	

•For	a	document	d and	a	class	c

`P(d/c)=P(c/d)*P(d)/ P(c)`


Naïve	Bayes	Classifier	(I)	

`MAP` is “maximum a posteriori” = most likely class 


`Cmap= argmax P(c|d) . (c belongs to C)`
    `= P(d/c)*P(c) `


https://web.stanford.edu/class/cs124/lec/naivebayes.pdf


