Title: Hierarchical Clustering
Date: 2021.01.22
Summary: Definition. Steps. From-Scrath Python implementation. Application on data sets of different shapes. Comparison with <em>k</em>-Means.


```python
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn import datasets

from collections import defaultdict
from collections.abc import Iterable
```


# Definition

Hierarchical Clustering is, in data mining and statistics, a method of cluster analysis which seeks to build a hierarchy of clusters. There are two types: 

* Agglomerative clustering 
* Divisive clustering.

A hierarchical cluster consists in a set of nested clusters that are arranged as a tree. Hierarchical methods are especially useful when the target is to arrange the clusters into a natural hierarchy. Unlike other algorithms such as *k*-means, these methods do not require the number of clusters as a parameter, and one can stop at the desired number of clusters. 

In order to decide which clusters should be combined (for agglomerative), or where a cluster should be split (for divisive), a **distance metric** and a **linkage criterion** are used as measures of dissimilarity between sets of observations.

## Distance Metric 

A distance metric must be chosen so that it defines similarity in a way that is sensible for the field of study. There are many distance metrics, e.g. Euclidean distance, Manhattan distance, Maximum distance, Mahalanobis distance. The choice of distance metric should be made based on theoretical concerns from the domain of study. For example, if clustering crime sites in a city, city block distance may be appropriate. Where there is no theoretical justification for an alternative, the Euclidean should generally be preferred, as it is usually the appropriate measure of distance in the physical world.

## Linkage Criteria

The linkage criteria determines how the distance between two clusters is computed.

Let $A$ and $B$ be two clusters, and $x_A$ and $x_B$ two point so that $x_A \in A, x_B \in B$.

### Single linkage

It uses the minimum of the distances between all observations of the two clusters.

$d(A, B) = \min \{ d(x_A, x_B) \}$

### Average linkage

It is the average of the distances of each observation of the two sets.

$d(A, B) = \sum_{i,j} \dfrac{d(x_A, x_B)}{|x_A| \cdot |x_B|} $

### Complete linkage

It uses the maximum distances between all observations of the two sets.

$d(A, B) = \max \{ d(x_A, x_B) \}$

### Ward's method

It minimizes the variance of the clusters being merged.

For more information, relate to [Scipy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

<div class="alert alert-success">
The choice of linkage criteria should be made based on theoretical considerations from the domain of application. A key theoretical issue is the cause of variation. For example, in archaeology, variation is expected to occur through innovation and natural resources, so working out if two groups of artifacts are similar may make sense based on identifying the most similar members of the cluster. Where there are no clear theoretical justifications for the choice of linkage criteria, Ward’s method is the sensible default.
</div>
    
# Agglomerative Clustering

Agglomerative Clustering is a type of Hierarchical Clustering in which each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. The main output of Hierarchical Clustering is a **dendrogram**, a tree  diagram that shows the hierarchical relationship between the clusters.

Agglomerative Clustering is useful to find clusters in data where other algorithms (such as *k*-means) fail, especially when the data is nonspherical or anisotropic, and the potencial clusters have unequal variance or different sizes. Nevertheless, it requires the computation and storage of an $n \times n$ matrix, which, for very large datasets, can be computationally expensive and slow, and its performance depends strongly on the distance metric and the linkage criterion chosen.

## Steps

1. Set each observation as a separate cluster.

*Repeat*

2. Compute the Cluster Distance Matrix. Each entry $i,j$ of the matrix is the distance between cluster$_i$ and cluster$_j$, according to a distance metric and linkage criteria.

3. Select clusters $A$ and $B$ as the cluster$_i$ and cluster$_j$ with minimum distance in the Cluster Distance Matrix.

4. Merge the clusters $A$ and $B$.

*Until there is one cluster left*

# Implementation

Agglomerative Clustering will be implemented using Euclidean Distance and Single Linkage Criterion.


```python
def hierarchical_agglomerative_clustering(X, num_clusters=2):
    
    # STEP 1: Create one cluster for each point
    clusters = defaultdict(list)
    for i in range(X.shape[0]):
        clusters[i].append(i)

    current_num_clusters = len(clusters)
    
    while current_num_clusters > 1: 

        # STEP 2: Calculate Cluster Distance Matrix
        
        # Create empty distance matrix
        cluster_distance_matrix = np.zeros((current_num_clusters, current_num_clusters))

        # Iterate over each element of cluster distance matrix
        for row_index in range(current_num_clusters):
            for column_index in range(current_num_clusters):
                
                # If clusters are different
                if row_index != column_index:

                    # Get points in each cluster
                    points_indexes_cluster_1 = np.array(flatten(clusters[row_index]))
                    points_indexes_cluster_2 = np.array(flatten(clusters[column_index]))

                    between_clusters_distances = []
                    
                    # Iterate over each point index of both clusters
                    for point_index_cluster_1 in points_indexes_cluster_1:
                        for point_index_cluster_2 in points_indexes_cluster_2:
                            
                            # Get both points
                            point_cluster_1 = X[point_index_cluster_1]
                            point_cluster_2 = X[point_index_cluster_2]
                            
                            # Calculate Euclidean Distance
                            distance = np.linalg.norm(point_cluster_1 - point_cluster_2)
                            
                            between_clusters_distances.append(distance)
                    
                    # Set cluster distance matrix entry according to Single Linkage Criterion
                    cluster_distance_matrix[row_index, column_index] = np.min(between_clusters_distances)

        # Diagonal is 0. Set it to infinity so that it is not chosen as minimum
        np.fill_diagonal(cluster_distance_matrix, np.inf)
        
        # STEP 3: Get indexes of the two clusters with minimum distance
        cluster_A, cluster_B = np.unravel_index(cluster_distance_matrix.argmin(), cluster_distance_matrix.shape)
        
        # STEP 4: Merge cluster A and B
        
        # Check number of clusters required
        if current_num_clusters == num_clusters:
            return clusters
        
        else:
            # If cluster A is a list of clusters, append cluster B to the last cluster as a sub list
            try:
                clusters[cluster_A][-1].append(clusters[cluster_B])
            
            # If cluster A is not a list of clusters, append cluster B to cluster A as a sub list
            except:
                clusters[cluster_A].append(clusters[cluster_B])
 
            # Remove cluster B
            clusters.pop(cluster_B)
            

        # Rename dictionary keys
        old_keys = list(clusters.keys())
        for new_key, old_key in enumerate(old_keys):
            clusters[new_key] = clusters.pop(old_key)

        # Update number of clusters
        current_num_clusters = len(clusters)
```


```python
# Auxiliary function to flatten lists of lists of lists...
def flatten(x):
    if isinstance(x,Iterable):return[C for B in x for C in flatten(B)]
    else:return[x]
```

# Application on Data Sets of Different Shapes

## Moons


```python
num_samples = 80
X = datasets.make_moons(n_samples=num_samples, noise=.05)[0]
```


```python
hac = hierarchical_agglomerative_clustering(X, num_clusters=2)
```


```python
labels = np.zeros(num_samples, dtype=int)
for i in range(len(hac)):
    cluster = flatten(hac[i])
    labels[cluster] = i
```


```python
plt.scatter(*X.T, color=np.choose(labels, ["darkcyan", "saddlebrown"]))
```


    
![image alt text]({static}../images/hierarchical_clustering_1.png)
    



```python
dendrogram(linkage(X, 'single'))
```


    
![image alt text]({static}../images/hierarchical_clustering_2.png)
    


**Comparirson with $k$-means**


```python
from sklearn.cluster import KMeans
```


```python
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

plt.scatter(*X.T, color=np.choose(kmeans.labels_, ["darkcyan", "saddlebrown"]))
```


    
![image alt text]({static}../images/hierarchical_clustering_3.png)
    


<div class="alert alert-warning">
Hierarchical Clustering require a certain amount of observations to perform correctly. Therefore, data sets of different shapes will be used to test the algorithm with Sklearn, as it is optimized to work with large data sets.
</div>


```python
from sklearn.cluster import AgglomerativeClustering
```

## Two Spirals


```python
np.random.seed(2)
n_points = 1500
noise = .5
n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi) / 360
d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))

clustering = AgglomerativeClustering(linkage='single').fit(X)
labels = clustering.labels_

plt.scatter(*X.T, color=np.choose(labels, ["darkcyan", "saddlebrown"]))
```


    
![image alt text]({static}../images/hierarchical_clustering_4.png)
    


## Contained Circle


```python
X = datasets.make_circles(n_samples=1500, factor=.5, noise=.05, random_state=2)[0]

clustering = AgglomerativeClustering(linkage='single').fit(X)
labels = clustering.labels_

plt.scatter(*X.T, color=np.choose(labels, ["darkcyan", "saddlebrown"]))
```


    
![image alt text]({static}../images/hierarchical_clustering_5.png)
    


## Different Variance Blobs


```python
X = datasets.make_blobs(n_samples=1500, n_features=2, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5], random_state=2)[0]

clustering = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
labels = clustering.labels_

plt.scatter(*X.T, color=np.choose(labels, ["darkcyan", "saddlebrown", "darkmagenta"]))
```


    
![image alt text]({static}../images/hierarchical_clustering_6.png)
    


# Hierarchical Divisive Clustering

The other Hierarchical Clustering algorithm is the Divisive Hierarchical Clustering, which initially groups all the observations into one cluster, and then successively splits these clusters until there is one cluster for each observation. It is rarely used in practice.

# References

[Hierarchical Clustering - Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_clustering){: target="_blank"}

[What is Hierarchical Clustering?
](https://www.displayr.com/what-is-hierarchical-clustering/#:~:text=Hierarchical%20clustering%2C%20also%20known%20as,broadly%20similar%20to%20each%20other){: target="_blank"}

[sklearn.cluster.AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering){: target="_blank"}

[sklearn Clustering](https://scikit-learn.org/stable/modules/clustering.html){: target="_blank"}

[Difference between K means and Hierarchical Clustering](https://www.geeksforgeeks.org/difference-between-k-means-and-hierarchical-clustering/){: target="_blank"}
