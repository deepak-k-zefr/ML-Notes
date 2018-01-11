## K-means

More Reading(https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)

k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster


K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:
1. The centroids of the K clusters, which can be used to label new data
2. Labels for the training data (each data point is assigned to a single cluster)

Assumptions:
1. k-means assumes the variance of the distribution of each attribute (variable) is spherical;
2. All variables have the same variance:

Assumption Fail ::

![fig](https://i.stack.imgur.com/tXGTo.png)




3. The prior probability for all k clusters is the same, i.e., each cluster has roughly equal number of observations;
4. There are K clusters



## K-means++

is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.
 
The k-means problem is to find cluster centers that minimize the intra-class variance, i.e. the sum of squared distances from each data point being clustered to its cluster center (the center that is closest to it). Although finding an exact solution to the k-means problem for arbitrary input is NP-hard, the standard approach to finding an approximate solution (often called Lloyd's algorithm or the k-means algorithm) is used widely and frequently finds reasonable solutions quickly.

However, the k-means algorithm has at least two major theoretic shortcomings:

First, it has been shown that the worst case running time of the algorithm is super-polynomial in the input size.[5]
Second, the approximation found can be arbitrarily bad with respect to the objective function compared to the optimal clustering.

The k-means++ algorithm addresses the second of these obstacles by specifying a procedure to initialize the cluster centers before proceeding with the standard k-means optimization iterations. With the k-means++ initialization, the algorithm is guaranteed to find a solution that is O(log k) competitive to the optimal k-means solution.


#### Example of a sub-optimal clustering

To illustrate the potential of the k-means algorithm to perform arbitrarily poorly with respect to the objective function of minimizing the sum of squared distances of cluster points to the centroid of their assigned clusters, consider the example of four points in R2 that form an axis-aligned rectangle whose width is greater than its height.

If k = 2 and the two initial cluster centers lie at the midpoints of the top and bottom line segments of the rectangle formed by the four data points, the k-means algorithm converges immediately, without moving these cluster centers. Consequently, the two bottom data points are clustered together and the two data points forming the top of the rectangle are clustered together—a suboptimal clustering because the width of the rectangle is greater than its height.

Now, consider stretching the rectangle horizontally to an arbitrary width. The standard k-means algorithm will continue to cluster the points suboptimally, and by increasing the horizontal distance between the two data points in each cluster, we can make the algorithm perform arbitrarily poorly with respect to the k-means objective function.

#### Improved initialization algorithm
The intuition behind this approach is that spreading out the k initial cluster centers is a good thing: the first cluster center is chosen uniformly at random from the data points that are being clustered, after which each subsequent cluster center is chosen from the remaining data points with probability proportional to its squared distance from the point's closest existing cluster center.

The exact algorithm is as follows:

1. Choose one center uniformly at random from among the data points.
2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
4. Repeat Steps 2 and 3 until k centers have been chosen.
5. Now that the initial centers have been chosen, proceed using standard k-means clustering.
6. This seeding method yields considerable improvement in the final error of k-means. Although the initial selection in the algorithm takes extra time, the k-means part itself converges very quickly after this seeding and thus the algorithm actually lowers the computation time. The authors tested their method with real and synthetic datasets and obtained typically 2-fold improvements in speed, and for certain datasets, close to 1000-fold improvements in error. In these simulations the new method almost always performed at least as well as vanilla k-means in both speed and error.

The k-means++ algorithm guarantees an approximation ratio O(log k) in expectation (over the randomness of the algorithm), where k is the number of clusters used. This is in contrast to vanilla k-means, which can generate clusterings arbitrarily worse than the optimum

### Applications
k-means clustering is rather easy to implement and apply even on large data sets, particularly when using heuristics such as Lloyd's algorithm . It has been successfully used in various topics, including market segmentation, computer vision, astronomy and agriculture. 
It often is used as a preprocessing step for other algorithms, for example to find a starting configuration.


### Why is Euclidean distance not a good metric in high dimensions?
The notion of Euclidean distance, which works well in the two-dimensional and three-dimensional worlds studied by Euclid, has some properties in higher dimensions that are contrary to our (maybe just my) geometric intuition which is also an extrapolation from two and three dimensions.

Consider a 4×44×4 square with vertices at (±2,±2)(±2,±2). Draw four unit-radius circles centered at (±1,±1)(±1,±1). These "fill" the square, with each circle touching the sides of the square at two points, and each circle touching its two neighbors. For example, the circle centered at (1,1)(1,1) touches the sides of the square at (2,1)(2,1) and (1,2)(1,2), and its neighboring circles at (1,0)(1,0) and (0,1)(0,1). Next, draw a small circle centered at the origin that touches all four circles. Since the line segment whose endpoints are the centers of two osculating circles passes through the point of osculation, it is easily verified that the small circle has radius r2=2‾√−1r2=2−1 and that it touches touches the four larger circles at (±r2/2‾√,±r2/2‾√)(±r2/2,±r2/2). Note that the small circle is "completely surrounded" by the four larger circles and thus is also completely inside the square. Note also that the point (r2,0)(r2,0) lies on the small circle. Notice also that from the origin, one cannot "see" the point (2,0,0)(2,0,0) on the edge of the square because the line of sight passes through the point of osculation (1,0,0)(1,0,0) of the two circles centered at (1,1)(1,1) and (1,−1)(1,−1). Ditto for the lines of sight to the other points where the axes pass through the edges of the square.

Next, consider a 4×4×44×4×4 cube with vertices at (±2,±2,±2)(±2,±2,±2). We fill it with 88 osculating unit-radius spheres centered at (±1,±1,±1)(±1,±1,±1), and then put a smaller osculating sphere centered at the origin. Note that the small sphere has radius r3=3‾√−1<1r3=3−1<1 and the point (r3,0,0)(r3,0,0) lies on the surface of the small sphere. But notice also that in three dimensions, one can "see" the point (2,0,0)(2,0,0) from the origin; there are no bigger bigger spheres blocking the view as happens in two dimensions. These clear lines of sight from the origin to the points where the axes pass through the surface of the cube occur in all larger dimensions as well.

Generalizing, we can consider a nn-dimensional hypercube of side 44 and fill it with 2n2n osculating unit-radius hyperspheres centered at (±1,±1,…,±1)(±1,±1,…,±1) and then put a "smaller" osculating sphere of radius
rn=n‾√−1(1)
(1)rn=n−1
at the origin. The point (rn,0,0,…,0)(rn,0,0,…,0) lies on this "smaller" sphere. But, notice from (1)(1) that when n=4n=4, rn=1rn=1 and so the "smaller" sphere has unit radius and thus really does not deserve the soubriquet of "smaller" for n≥4n≥4. Indeed, it would be better if we called it the "larger sphere" or just "central sphere". As noted in the last paragraph, there is a clear line of sight from the origin to the points where the axes pass through the surface of the hypercube. Worse yet, when n>9n>9, we have from (1)(1) that rn>2rn>2, and thus the point (rn,0,0,…,0)(rn,0,0,…,0) on the central sphere lies outside the hypercube of side 44 even though it is "completely surrounded" by the unit-radius hyperspheres that "fill" the hypercube (in the sense of packing it). The central sphere "bulges" outside the hypercube in high-dimensional space. I find this very counter-intuitive because my mental translations of the notion of Euclidean distance to higher dimensions, using the geometric intuition that I have developed from the 2-space and 3-space that I am familiar with, do not describe the reality of high-dimensional space.

what is 'high dimensions'?" is n≥9n≥9.


## GMM

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.

a mixture model is a probabilistic model for representing the presence of subpopulations within an overall population, without requiring that an observed data set should identify the sub-population to which an individual observation belongs.

[PROS:](http://scikit-learn.org/stable/modules/mixture.html#pros)
[CONS:](http://scikit-learn.org/stable/modules/mixture.html#cons)

### Estimation algorithm Expectation-maximization
The main difficulty in learning Gaussian mixture models from unlabeled data is that it is one usually doesn’t know which points came from which latent component (if one has access to this information it gets very easy to fit a separate Gaussian distribution to each set of points). Expectation-maximization is a well-founded statistical algorithm to get around this problem by an iterative process. First one assumes random components (randomly centered on data points, learned from k-means, or even just normally distributed around the origin) and computes for each point a probability of being generated by each component of the model. Then, one tweaks the parameters to maximize the likelihood of the data given those assignments. Repeating this process is guaranteed to always converge to a local optimum.

## DBSCAN, (Density-Based Spatial Clustering of Applications with Noise),
captures the insight that clusters are dense groups of points. The idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.

It works like this: First we choose two parameters, a positive number epsilon and a natural number minPoints. We then begin by picking an arbitrary point in our dataset. If there are more than minPoints points within a distance of epsilon from that point, (including the original point itself), we consider all of them to be part of a "cluster". We then expand that cluster by checking all of the new points and seeing if they too have more than minPoints points within a distance of epsilon, growing the cluster recursively if so.

Eventually, we run out of points to add to the cluster. We then pick a new arbitrary point and repeat the process. Now, it's entirely possible that a point we pick has fewer than minPoints points in its epsilon ball, and is also not a part of any other cluster. If that is the case, it's considered a "noise point" not belonging to any cluster.
