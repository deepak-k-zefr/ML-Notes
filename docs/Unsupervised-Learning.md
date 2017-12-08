## K-means

More Reading(https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)

k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster


K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:
1. The centroids of the K clusters, which can be used to label new data
2. Labels for the training data (each data point is assigned to a single cluster)

Assumptions:
1. k-means assumes the variance of the distribution of each attribute (variable) is spherical;
2. all variables have the same variance:



![fig](https://i.stack.imgur.com/tXGTo.png)




3. the prior probability for all k clusters is the same, i.e., each cluster has roughly equal number of observations;
4. There are K clusters

Applications
k-means clustering is rather easy to implement and apply even on large data sets, particularly when using heuristics such as Lloyd's algorithm . It has been successfully used in various topics, including market segmentation, computer vision, astronomy and agriculture. 
It often is used as a preprocessing step for other algorithms, for example to find a starting configuration.



## GMM
