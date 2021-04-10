Title: <em>k</em>-Nearest Neighbors
Date: 2021-01-20
Summary: Definition. Assumptions. Finding of optimum *k* with K-folds Cross-validation. Testing of different distance metrics, including Euclidean, Manhattan and Cosine. Application and analysis on Iris dataset.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from scipy.spatial import distance_matrix
from scipy.spatial import distance
```


# 1 Definition

*k*-nearest neighbors algorithm (*k*-NN) is a classification method developed in 1951. It is a supervised machine learning algorithm used for both classification and regression, although in this work the classification approach will be studied.

The observed data is the input of the algorithm, and it consists of the all the observed data points and their labels. 

## 1.1 Assumptions

* Similar things exist in close proximity, i.e., similar things are near to each other. The data is not randomly spread.

* The data is in a feature space (abstract space where each sample is represented as a point in n-dimensional space), in which exists a notion of distance.

* No assumption on the underlying data distribution is made.

## 1.2 Steps

1. Calculate the distance from each new point to the observed points.

2. Select the *k* points with minimum distance.

3. Set the mode (the most frequent) label of the *k* points as the estimation or prediction for the new point.

## 1.3 Considerations

* To improve the algorithm performance, it is recommended to scale the data (between 0 and 1).

* Although euclidean distance is used the most, other distance metrics such as Canberra, Mahalanobis, Manhattan, Minkowski, Chebyshev, Cosine can be implemented. For more information, read [Abu Alfeilat, Haneen Arafat, et al. "Effects of distance measure choice on k-nearest neighbor classifier performance: a review." (2019)](https://arxiv.org/pdf/1708.04321.pdf).

* It is considered a *lazy* algorithm, meaning that it does not learn anything in the training period. The algorithm learns only at the time of making real time predictions.

* If the new point is considerably far apart from the observed data, *k*-NN can be used to classify it, but it is probable that the classification is wrong. For preventing such cases, a limit to the distance from a new point to the rest of the data can be set.

* *k*-NN should not classify points for which two or more labels are equally possible.

* $k=1$ is often called *1-nearest neighbor*, and it assigns the new point the label of the nearest point. It might lead to overfitting.

* In large or high dimensions datasets, it is very expensive to calculate the distances between all points/dimensions.

* It is sensitive to noisy data, missing values and outliers.

# 2 Implementation


```python
def k_nearest_neighbors(k, train_x, train_y, test_x, metric='euclidean'):
    '''
    Train_x can be considered as the training points (cross-validation) or the original data (prediction)
    Train_y can be considered as the training labels of the orginal points (cross-validation) or the labels of the original data (prediction)
    Test_x can be considered as the testing points (cross-validation) or the new points (prediction)    
    '''
    
    # Scale data between 0 and 1 (min-max normalization)
    train_x_min = train_x.min(axis=0)
    train_x_max = train_x.max(axis=0)
    train_x = (train_x - train_x_min) / (train_x_max - train_x_min)
    test_x = (test_x - train_x_min) / (train_x_max - train_x_min)
    
    # Calculate distance from each test point to train points
    distances_matrix = distance.cdist(test_x, train_x, metric=metric)
    threshold = np.quantile(distances_matrix, 0.05) 
    
    estimations = []
    for distances in distances_matrix:
        
        # Get k train points of minimum distances
        indexes = np.argsort(distances)[0:k]
        
        # Minimum distance restriction
        # If the minimum distance is greater than 5% of the distances, no prediction can be done
        # Threshold should be chosen according to the data domain
        shortest_distance = distances.min()
        if shortest_distance >= threshold:
            estimations.append(3)
            continue
        
        # Get label corresponding to each train point
        unique, counts = np.unique(train_y[indexes], return_counts=True)
        
        # Two-label restriction
        # When two labels have same frequency, no prediction can be done
        if len(counts) > 1:
            first, second, *_ = np.sort(counts)
            if first == second:
                estimations.append(3)
                continue
        
        # Get estimation for test point
        estimation = unique[np.argmax(counts)]
        estimations.append(estimation)
      
    return np.array(estimations)
```

# 3 Application

## 3.1 Dataset

The [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) will be used to exemplify the use of the *k*-NN algorithm. 


```python
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
```


```python
colors = ['mediumpurple', 'lightsalmon', 'lightskyblue', 'darkgrey']

plt.scatter(*X.T, color=np.choose(y.astype(int), colors))

```


    
![image alt text]({static}../images/k_nearest_neighbors_1.png)
    


## 3.2 Training Process

The *k*-NN algorithm will be trained using K-folds Cross-validation and accuracy (percentage of successes) as the error measure. K-folds will be run for 30 values of *k*. For each *k*, the accuracy for each fold will be recorded. The optimum *k* value will be set as the one which presented the highest accuracy median for every K.

(Note that *k* refers to *k*-NN, and K refers to Cross-validation)

<br>

**Train-Validation Split**

```python
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
```

**K-folds Cross-validation**


```python
metric = 'euclidean'

folds = 7
kf = KFold(n_splits=folds)

k_values = np.arange(1, 31)

accuracy_list_training = []

for k in k_values:
    
    folds_accuracy = []
    
    for train_index, test_index in kf.split(X_train):
        
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        estimations = k_nearest_neighbors(k, X_train_fold, y_train_fold, X_test_fold, metric=metric)
        
        accuracy = np.mean(estimations == y_test_fold)
        folds_accuracy.append(accuracy)
        
    accuracy_list_training.append(folds_accuracy)
```

### 3.2.2 Optimum *k*

Boxplots for the accuracy for each *k* considering all K folds will be constructed. The optimum *k* will be the one with maximum median. The median was used instead of the mean to minimize the effects of outliers.


```python
accuracy_training_df = pd.DataFrame(data=accuracy_list_training, index=k_values)

best_k_training = np.argmax(accuracy_training_df.median(axis=1)) + 1
best_accuracy_training = np.max(accuracy_training_df.median(axis=1))

print(f'The value of k that presented the best accuracy median in {folds} folds is: {best_k_training} with value of {best_accuracy_training:.3g}')

ax = sns.boxplot(data=accuracy_training_df.T, color='maroon')
```

    The value of k that presented the best accuracy median in 7 folds is: 3 with value of 0.706
    


    
![image alt text]({static}../images/k_nearest_neighbors_2.png)
    


## 3.3 Test Validation Set

Once the optimum *k* was found using Cross-validation, the production environment will be tested using the validation set.


```python
estimations = k_nearest_neighbors(best_k_training, X, y, X_validation, metric=metric)

accuracy_validation = np.mean(estimations == y_validation)

print(f'Validation Set Accuracy for k={best_k_training}: {accuracy_validation:.3g}')
```

    Validation Set Accuracy for k=3: 0.967
    

# 4 Distance Metrics Comparison

The training and validation procedure was performed for several distance metrics, and the results are illustrated in the following table, order from best to worst performance in validation set:

<br>

<div>
<table border="1">
<thead>
  <tr>
    <th class="tg-c3ow">Metric</th>
    <th class="tg-c3ow">Optimum <span style="font-style:italic">*k*</span></th>
    <th class="tg-c3ow">Training</th>
    <th class="tg-c3ow">Validation</th>
  </tr>
</thead>
  <tbody>
    <tr>
    <td class="tg-c3ow">Canberra</td>
    <td class="tg-c3ow">9</td>
    <td class="tg-c3ow">0.706</td>
    <td class="tg-c3ow">0.667</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mahalanobis</td>
    <td class="tg-c3ow">25</td>
    <td class="tg-c3ow">0.706</td>
    <td class="tg-c3ow">0.667</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Cityblock</td>
    <td class="tg-c3ow">29</td>
    <td class="tg-c3ow">0.765</td>
    <td class="tg-c3ow">0.667</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Euclidean</td>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">0.706</td>
    <td class="tg-c3ow">0.667</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Minkowski</td>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">0.706</td>
    <td class="tg-c3ow">0.667</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Cosine</td>
    <td class="tg-c3ow">17</td>
    <td class="tg-c3ow">0.706</td>
    <td class="tg-c3ow">0.533</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Chebyshev</td>
    <td class="tg-c3ow">13</td>
    <td class="tg-c3ow">0.706</td>
    <td class="tg-c3ow">0.5</td>
  </tr>
  </tbody>
</table>
</div>

<br>

As can be observed, the five first distances performed similar in both Training and Validation set, with Manhattan performing better in the Training set. The worst distances were Cosine and Chebyshev.

# 5 Prediction Map

To provide an estimate of the classification of different points in the observed data domain, prediction maps for 4 *k* values (including the optimum) will be made.


```python
xs = np.linspace(4, 8, 200)
ys = np.linspace(1.5, 5, 175)

X_space, Y_space = np.meshgrid(xs, ys)
X_space = np.ndarray.flatten(X_space)
Y_space = np.ndarray.flatten(Y_space)

points = np.dstack([X_space, Y_space])[0]

ks = [9, 29, 3, 17]

fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharex=True, sharey=True)

for ax, k, metric in zip(axes.flatten(), ks, ['Canberra', 'Cityblock', 'Euclidean', 'Cosine']):
    predictions = k_nearest_neighbors(k, X, y, points, metric).astype(int)
    ax.scatter(*points.T, color=np.choose(predictions, colors), s=10, alpha=0.4)
    ax.scatter(*X.T, color='black', s=50)
```


    
![image alt text]({static}../images/k_nearest_neighbors_3.png)
    
<br>


As can be observed, Manhattan and Euclidean distance provide the best predictions for all points. The cosine distance might be useful when the divisions are not Cartesian but angular and fails to classify the data in the present data set. Moreover, the Manhattan distance has horizontal or vertical lines as boundaries.

Also, it can be seen that adding the minimum distance and tie restriction, although not part of the original algorithm, makes the predictions more real, as the algorithm cannot predict points that are far enough from the rest of the data, as the basic assumption of *k*-NN is that similar points are close to each other, and a point far apart may belong to a new category not yet observed nor labeled. Furthermore, the two-labels restriction prevents the algorithm from classifying points for which two or more labels are equally possible. It can be noticed that, for all distances metrics, all optimum *k* values are odd numbers, meaning that these "cannot predict" areas are unlikely to exist between two labeled areas. This gray areas can be seen instead in the intersection between three labeled areas, as depicted in the Canberra and Euclidean cases.

# References

[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm){: target="_blank"}

[Machine Learning Basics with the K-Nearest Neighbors Algorithm
](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761){: target="_blank"}

[Why is KNN not “model-based”?
](https://stats.stackexchange.com/questions/81240/why-is-knn-not-model-based){: target="_blank"}

[Advantages and Disadvantages of KNN](http://theprofessionalspoint.blogspot.com/2019/02/advantages-and-disadvantages-of-knn.html){: target="_blank"}

[KNN: Failure cases, Limitations](https://levelup.gitconnected.com/knn-failure-cases-limitations-and-strategy-to-pick-right-k-45de1b986428){: target="_blank"}