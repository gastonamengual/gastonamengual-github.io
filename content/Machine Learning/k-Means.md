Title: <em>k</em>-means
Date: 2021.3.19
Summary: Definition. Assumptions. Steps. From-Scrath Python implementation. Application on dummy data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist
```


# Definition

*k*-means is a method of vector quantization (originally from signal processing) that aims to partition $n$ observations into $k$ clusters, in which each observation belongs to the cluster with the nearest mean (cluster centers or centroids). It is however most commonly defined as an unsupervised machine learning algorithm for clustering. 

### Characteristics

As a partitional clustering algorithm, it has the following characteristics:

* It divides data objects into non-overlapping groups: no object can be a member of more than one cluster.
* Every cluster must have at least one object.
* It finds a grouping so that similar objects are in the same cluster and dissimilar objects are in different clusters.


### Assumptions

*k*-means should be applied to data sets that meet the following conditions:

* Clusters are spatially grouped, convex, spherical, and isotropic.
* Clusters are of a similar size and density.
* There are no outliers.

### Model

Let X be an $n \times m$ matrix, with $n$ observations vectors of $m$ features. The algorithm divides a $X$ into $k$ disjoint clusters $C_j$ $(j=1,2,\dotsc,k)$, each described by the mean or centroid $\mu_j$ of the samples in the cluster. These centroids are chosen so that they minimize the **within-cluster sum-of-squares (WCSS)** criterion. Namely,

$$\sum_{i=1}^{n} \underset{\mu_j \in C_j}{\min} = (||x_i - \mu_{j}||)^2$$

The optimization problem then consists on minimizing the sum of the norm (distance) of the vector that results from the difference of each centroid and its its assigned samples. 

### Steps

The most basic *k*-means algorithm is called Lloyd's algorithm. It requires a number $k$ of clusters to assign as a parameter, and it has the following steps:

###

$\quad$ **1.** Randomly initialize $k$ centroids, either as $k$ random samples from the data, or as $k$ random points in the domain of the data.

###

*repeat*

###

$\quad$ **2.** Assign each point to its nearest centroid.
    
$\quad$ **3.** Create the new centroids as the mean of the samples assigned to each previous centroid.

$\quad$ **4.** Calculate the difference between the old and the new centroids.

###

*until convergence*.

### Initialization

Given enough time, K-means will always converge. However, depending on the initialization of the centroids, this may be to a local minimum. As a result, the computation is often done several times, with different initializations of the centroids. One method to help address this issue consists on initializing the centroids to be (generally) distant from each other, leading to provably better results than random initialization.

### Convergence

Convergence is reached when the difference between old and new centroids is less than a threshold $\epsilon$, meaning that the centroids do not move significantly.

# Implementation


```python
def k_means(data, k, epsilon=1e-4, max_iterations=1000):
    
    # Step 1 - Initialize centroids as random sample points
    centroids = data[np.random.randint(data.shape[0], size=k)]
    
    centroids_list = [centroids]

    for _ in range(max_iterations):
        
        # Step 2
        # Calculate distances from each point to all centroids
        distances = distance_matrix(data, centroids)
        
        # Calculate nearest centroid to each point
        estimations = np.argmin(distances, axis=1)

        previous_centroids = centroids.copy()
        
        # Step 3 - Calculate new centroid as mean of samples
        centroids = np.array([np.mean(data[estimations == i], axis=0) for i in range(k)])
        centroids_list.append(centroids)
        
        # Step 4 - Check for convergence
        difference_between_previous_absolute = np.linalg.norm(centroids - previous_centroids, axis=1)
        difference_between_previous_relative = difference_between_previous_absolute / np.linalg.norm(previous_centroids, axis=1)
        
        if np.mean(difference_between_previous_relative) < epsilon:
            break
            
    return estimations, centroids, np.array(centroids_list)
```


```python
def calculate_wcss(data, estimations, centroids, k):
    wcss = 0
    for i in range(k):
        norm = np.linalg.norm(data[estimations == i] - centroids[i])
        norm_squared = norm**2
        wcss += norm_squared
    return wcss
```

# Application

### Create Dummy Data Set


```python
num_samples = 1000
data, _ = make_blobs(n_samples=num_samples, centers=3, n_features=2, random_state=9)
```


```python
plt.scatter(*data.T)
```


    
![image alt text]({static}../images/k_means_1.png)
    


### Find optimum *k*

In order to choose the optimum value of clusters, the WCSS will be calculated for different values of *k*.


```python
ks = np.arange(1, 7)
wcss_list = []

for k in ks:
    estimations, centroids, centroids_list = k_means(data, k)
    wcss = calculate_wcss(data, estimations, centroids, k)
    wcss_list.append(wcss)
```


```python
plt.plot(ks, wcss_list)
```


    
![image alt text]({static}../images/k_means_2.png)


From applying the elbow method, it can be observed that $k=3$ results in a lower WCSS than $k=1$ and $k=2$, and increasing the number of clusters from $k=3$ does not improve the WCSS significantly, for what it is not necessary to add more clusters. It is concluded than the optimum value of clusters is $k=3$. 

### Run *k*-means with optimum *k*


```python
optimum_k = 3
estimations, centroids, centroids_list = k_means(data, optimum_k)
wcss = calculate_wcss(data, estimations, centroids, optimum_k)
```


```python
colors = ["darkmagenta", "saddlebrown", "darkcyan"]

estimated_colors = np.choose(estimations, colors)

plt.scatter(*data.T, color=estimated_colors)
plt.scatter(*centroids.T, color=colors[:optimum_k])

for i in range(optimum_k):
    plt.plot(*centroids_list[:, i].T)
```

    
![image alt text]({static}../images/k_means_3.png)
    


The algorithm has successfully divided the data into three clusters.

# Observations

The 2 dimensional application described is useful to provide a simple understanding of the algorithm, of how to find the optimum *k*, and how to evaluate the performance of the algorithm. However, in real application, it is hardly dealt with 2D data sets where the clusters are clearly visible or detectable using edge detection methods. The goal of clustering algorithms is to find clusters when humans cannot, like, for example, in cases with multidimensional data.

<hr>

The measure to evaluate the performance of the algorithm is the value of the function that is minimized. That function only applies when the data meets the assumptions explained above. However, what happens if the data does not meet those assumptions (it is not convex, the clusters have different sizes, it presents outliers, clusters are superposed), or when it is not possible to know for certain its shape? For those cases, SSE will not be an effective performance measure. For those cases, it would be then preferable to measure the performance of the algorithm by analyzing the points assigned to the clusters created in many replicates. This way, even if the above criteria is not met, either because of unawareness or because it cannot be known (e.g. many dimensions), the measure would give more precise information on the accuracy of the results.

# References

[*k*-means clustering - Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)

[K-Means Clustering in Python: A Practical Guide](https://realpython.com/k-means-clustering-python/#what-is-clustering)

[scikit learn - Clustering - K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)

[K-means Clustering Python Example](https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203)

[Partitional Clustering](http://cs-people.bu.edu/evimaria/Italy-2015/partitional-clustering.pdf)

[K-Means Clustering in Python: A Practical Guide](https://realpython.com/k-means-clustering-python/#what-is-clustering)

[Assumptions Can Ruin Your K-Means Clusters](https://blog.learningtree.com/assumptions-ruin-k-means-clusters/)

[K-means clustering is not a free lunch](http://varianceexplained.org/r/kmeans-free-lunch/)
