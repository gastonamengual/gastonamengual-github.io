Title: Decision Trees
Date: 2021.03.30
Summary: Definition. General steps. Decision tree classifier implementation from scratch. Application of classifier and regressor. Definition of pruning.

### Table of Contents

[1 Definition](#Definition)

[2 General Steps](#General-Steps)

[3 Decision Tree Classifier](#Decision-Tree-Classifier)

[4 Decision Tree Regressor](#Decision-Tree-Regressor)

[5 Pruning](#Pruning)


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from pprint import pprint
```


<h1 id="Definition">1 Definition</h1>

A decision tree is a predictive modeling approach used in statistics, data mining, and machine learning to go from observations about an item to conclusions about the item's target value, and to visually and explicitly represent decisions, and to describe data. 

There two major types of decision tree models, integrated by the name of **CART** (Classification And Regression Tree). In **classification trees**, the predicted outcome is a discrete class to which the data belongs. In **regression trees**, the predicted outcome is a real number.

A decision tree consists of:

* Root: it indicates the entire data sample.
* Nodes: indicate two possible outcomes according to a certain probability, and are split into two nodes, one for each result.
* Leaves: indicate a final outcome, the "decision" taken after visiting all previous nodes.

<h1 id="General-Steps">2 General Steps</h1>

**1.** Validate the stop conditions (number of unique labels, minimum number of samples, tree maximum depth). 

*If one condition is met, stop the algorithm and return the prediction. If no condition is met, follow step 2.*

**2.** Iterate over each feature and each unique value of the feature. For each pair, split the labels in two according to the following rule: 

$$\text{The left split contains the labels of the points smaller or equal than the value of the feature.}$$ 
$$\text{The right split contains the labels of the points bigger than value of the feature.}$$

**3.** Choose the best feature and value according to some metric (see Section 3.1 and 4.1).

**4.** Split the data into two branches, left and right, using the best feature and value.

*Repeat Steps 1, 2, 3 and 4 for both branches.*

<h1 id="Decision-Tree-Classifier">3 Decision Tree Classifier</h1>


In classification decision trees, the target variables are categorical or discrete. Categorical variables contain a finite number of categories or distinct groups, which might not have a logical order (e.g. gender, material type, and payment method). Discrete variables are numeric variables that have a countable number of values between any two values (e.g. the number of customer complaints or the number of flaws or defects).

The data input of a decision tree is defined as an $m \times n$ feature matrix $X$, containing $m$ samples described by $n$ features, and an $m \times 1$ vector $y$ that contains the corresponding categorical or discrete property values (label or classes) to the samples in $X$.

## 3.1 Metrics

The best feature and value in which to split the data in Classification Decision Trees is decided according to a metric. These metrics include Entropy, Gini impurity, Variance reduction, and Measure of Goodness.

### 3.1.1 Entropy

In information theory, the entropy of a random variable is the average level of information or uncertainty inherent in the variable's possible outcomes. It was introduced by Claude Shannon in 1948. Given a discrete random variable $X$ with possible outcomes $x_1, x_2, \dotsc, x_n$ which occur with probabilities $P(x_1), P(x_2), \dotsc, P(x_n)$, the entropy of $X$ is defined as:

$$H(X) = - \sum_{i=1}^{n} P(x_i) \text{log} P(x_i)$$

The choice of the logarithm's base $b$ defines the unit: bits or shannons for $b=2$, nats for $b=e$, bans for $b=10$. Let $X$ be a random variable such that $x \sim Binomial(1, p)$ (Bernoulli). $X$ can have only two possible outcomes, the first with probability $p$ and the second with probability $1-p$. Then, the entropy can be written as:

$$H(p) = -p \cdot \text{log}_2(p) - (1 - p) \cdot \text{log}_2(1 - p)$$

Computing the derivative of $H(p)$ and setting it equal to $0$, the global maximum is found at $p = 0.5$, point at which the entropy is maximized with value $1$, as both outcomes are equally possible and the level of information is maximum. In $p=0$ and $p=1$, however, the entropy is $0$, as it is known beforehand which value will be the outcome. 


![image alt text]({static}../images/decision_trees_1.png)
    


**Overall Entropy**

In decision trees, a leaf (that is, a group in which all samples belong to one class) has minimum impurity, a probability of $1$, and an entropy of $0$. A node with 50% of each class has maximum impurity, a probability of $0.5$, and an entropy of $1$. The entropy increases as the number of classes increases. The Overall Entropy is calculated as the sum of the entropy of both splits:

$$H_{\text{overall}} = \sum_{j=1}^2 p_j H(p_j)$$

The best overall entropy will be the minimum value found in all splits made, as the associated uncertainty with making predictions must be the lowest possible.

**Information gain**

It is another metric used to calculate how important a feature is, and to decide the ordering of features in the nodes of the decision tree.

$$\text{Information gain} = \text{entropy}_{\text{parent}} - \text{weighted average entropy}_{\text{children}}$$

For more information, read [Information gain in decision trees](https://www.wikiwand.com/en/Information_gain_in_decision_trees).

### 3.1.2 Gini impurity

Gini impurity is a measure of how often  a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. For a set of items with $J$ classes, let $p_i$ with $i \in \{1, 2, \dotsc, J\}$ be the fraction of items labeld with class $i$ in the set, the Gini impurity is defined as

$$I_G(p) = 1 - \sum_{i=1}^{J} p_i^2$$

A Gini Impurity of 0 is the lowest and best possible impurity. It can only be achieved when everything is the same class.

## 3.2 Implementation

The algorithm implemented considers both continuous/discrete and categorical features, and two stopping conditions (minimum number of samples - maximum tree depth).


```python
def decision_tree_classifier(X, y, features_name, labels, min_samples=1, max_depth=None, depth_count=0):
    
    # Transpose X: one vector for each feature
    X = X.T
    
    # STEP 1: Validate Stopping Conditions
    # No more splits can be done - Min number of samples reached - Maximum depth reached
    if (np.unique(y).size == 1) or (X.shape[1] < min_samples) or (depth_count == max_depth):
        y_label_index = int(np.unique(y)[0])
        leaf = labels[y_label_index]
        return leaf
    
    depth_count += 1
    
    # Unique values for each feature
    unique_per_feature = [np.unique(feature) for feature in X]

    # Set best entropy as infinity
    overall_entropy = np.inf
    
    # STEP 2: Iterate over each feature and each unique value of the feature
    for feature_index, split_points in enumerate(unique_per_feature):
        
        feature = X[feature_index]
        
        # Iterate over all split points of one feature
        for split_point in split_points:

            # Split y in two
            left_split_indexes = feature <= split_point
                
            left_split, right_split = y[left_split_indexes], y[~left_split_indexes]

            # Calculate overall entropy
            probabilities_per_split = np.array([len(left_split), len(right_split)]) / len(feature)
            counts_per_split = [np.unique(data, return_counts=True)[1] for data in (left_split, right_split)]
            entropies_per_split = [stats.entropy(count, base=2) for count in counts_per_split]

            current_overall_entropy = probabilities_per_split @ entropies_per_split
            
            # STEP 3: Choose the best feature and value
            # Check best overall entropy
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_pair = (feature_index, split_point)
            
    # STEP 4
    # Split data according to best feature and best split point
    best_feature_index, best_split_point = best_pair
    
    left_split_indexes = X[best_feature_index] <= best_split_point 
        
    left_split_X, right_split_X = X.T[left_split_indexes], X.T[~left_split_indexes]
    left_split_y, right_split_y = y[left_split_indexes], y[~left_split_indexes]
    
    # Find left and right branches
    left_branch = decision_tree_classifier(left_split_X, left_split_y, features_name, labels, min_samples, max_depth, depth_count)
    right_branch = decision_tree_classifier(right_split_X, right_split_y, features_name, labels, min_samples, max_depth, depth_count)
    
    # Make decision tree
    feature_name = features_name[best_feature_index]
    decision_tree = {f'{feature_name} ≤ {best_split_point}': [left_branch, right_branch]}
    
    return decision_tree
```

Once the tree is built, predictions can be done for a certain new point. For that, the decision tree must be searched, following a specific branch depending on the satisfaction of the conditions.


```python
def classify_point(point, decision_tree, features_names):
    
    question = list(decision_tree.keys())[0]
    feature_name, comparison_operator, value = question.split()     
    
    feature_index = np.argwhere(features_names == feature_name)[0][0]
    
    if point[feature_index] <= float(value):
        answer = decision_tree[question][0]
    else:
        answer = decision_tree[question][1]
            
    if not isinstance(answer, dict):
        return np.argwhere(labels == answer)[0][0]
    else:
        return classify_point(point, answer, features_names)
```

## 3.3 Application


```python
iris = datasets.load_iris()

X = iris.data
features_names = np.array(['sepal_length','sepal_width','petal_length','petal_width','label'])

y = iris.target
labels = np.array(['setosa', 'versicolor', 'virginica'])

train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size=0.1333, random_state=42)
```

### Train Decision Tree


```python
decision_tree = decision_tree_classifier(train_set_X, train_set_y, features_names, labels)
pprint(decision_tree)
```

    {'petal_width ≤ 0.6': ['setosa',
                           {'petal_width ≤ 1.7': [{'petal_length ≤ 4.9': [{'petal_width ≤ 1.6': ['versicolor',
                                                                                                 'virginica']},
                                                                          {'petal_width ≤ 1.5': ['virginica',
                                                                                                 {'petal_length ≤ 5.1': ['versicolor',
                                                                                                                         'virginica']}]}]},
                                                  {'petal_length ≤ 4.8': [{'sepal_width ≤ 3.0': ['virginica',
                                                                                                 'versicolor']},
                                                                          'virginica']}]}]}
    

## 3.4 Visualization

For visualizing the tree, Sklearn's plot_tree will be used. <strong><em>value</em></strong> represents how many samples at that node fall into each category, it adds up to <strong><em>samples</em></strong>, which represents the number of samples at that node. The prediction corresponds to the most common category.


```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

```python
plot_tree(DecisionTreeClassifier(criterion='entropy', random_state=3).fit(X, y))
```


    
![image alt text]({static}../images/decision_trees_2.png)
    


### Testing


```python
predictions = [classify_point(point, decision_tree, features_names) for point in test_set_X]
accuracy = np.mean(predictions == test_set_y)
```

    Accuracy: 1.0
    

### Predictions


```python
fig, axes = plt.subplots(3, 2, figsize=(16, 7))
pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

for ax, pair_index in zip(axes.flatten(), pairs):

    pair = X[:, pair_index]
       
    # Train decision tree
    clf = DecisionTreeClassifier(criterion='entropy', random_state=3).fit(pair, y)

    x_min, x_max = pair[:, 0].min() - 1, pair[:, 0].max() + 1
    y_min, y_max = pair[:, 1].min() - 1, pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # Calculate and plot decision map
    Z = clf.predict(np.column_stack((xx.flatten(), yy.flatten())))
    Z = Z.reshape(xx.shape)
    cs = ax.contourf(xx, yy, Z)

    # Plot pair points
    ax.scatter(*pair.T)
```


    
![image alt text]({static}../images/decision_trees_3.png)
    


<h1 id="Decision-Tree-Regressor">4 Decision Tree Regressor</h1>

Decision trees where the target variable can take continuous values are called regression trees. Continuous variables are numeric variables that have an infinite number of values between any two values. A continuous variable can be numeric or date/time. For example, the length of a part or the date and time a payment is received.

The data input of a decision tree is defined as an $m \times n$ feature matrix $X$, containing $m$ samples described by $n$ features, and an $m \times 1$ vector $y$ that contains the values of the dependent variable.

## 4.1 Metrics

There are several metrics that can be used to split the data, such as MSE, MAE, Friedman MSE, and Poisson deviance.

#### MSE

One metric that can be used to split the data is the MSE. The data is split in two branches, and the MSE of each branch is calculated, taking the *true y* as the mean of the dependent variable, and the *predicted y* as the values of the points in the branch. Finally, the Total MSE is computed as the sum of the MSE of both branches. The RMSE can be likewise used.

## 4.2 Application

### Create dummy data set


```python
np.random.seed(1)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).flatten()
y[::5] += 3 * (0.5 - np.random.rand(40))

features_names = np.array(['x', 'y'])
```

### Train Decision Tree


```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
```

```python
plot_tree(DecisionTreeRegressor(max_depth=2, random_state=3).fit(X, y))
```


    
![image alt text]({static}../images/decision_trees_4.png)
    


### Find Optimum Depth

**Test Data**


```python
X_test = np.linspace(0.0, 5.0, 200).reshape(-1, 1)
```

**Train Decision Tree and calculate RMSE for different depths on Train and Test Data** 


```python
depths = np.arange(1, 19, 1)
rmses_train = []
rmses_test = []

for depth in depths:
    
    # RMSE on train data
    y_predicted = DecisionTreeRegressor(max_depth=depth, random_state=3).fit(X, y).predict(X)
    rmse = mean_squared_error(y, y_predicted, squared=False)
    rmses_train.append(rmse)
    
    # RMSE on test data
    y_predicted = DecisionTreeRegressor(max_depth=depth, random_state=3).fit(X, y).predict(X_test)
    rmse = mean_squared_error(y, y_predicted, squared=False)
    rmses_test.append(rmse)
```


```python
plt.plot(depths, rmses_test)
plt.plot(depths, rmses_train)
plt.scatter(np.argmin(rmses_test) + 1 , np.min(rmses_test))
```


    
![image alt text]({static}../images/decision_trees_5.png)
    


It can be observed that, as the error on the train set decreases towards 0, the error on the test set increases. This is due to an overfitting produced by high values of maximum depth. The optimum depth was chosen as that with the minimum RMSE on the test data.

### Predictions


```python
fig, axes = plt.subplots(2, 2)

depths = [1, 3, 10, 20]

for ax, depth in zip(axes.flatten(), depths):

    y_predicted = DecisionTreeRegressor(max_depth=depth, random_state=3).fit(X, y).predict(X_test)

    ax.scatter(X, y)
    ax.plot(X_test, y_predicted)

```


    
![image alt text]({static}../images/decision_trees_6.png)
    


<h1 id="Pruning">5 Pruning</h1>

Pruning is a data compression technique that reduces the size of a decision tree by removing sections of the tree that are non-critical or redundant to classify instances, reducing the complexity of the final classifier and improving predicting accuracy by the reduction of overfitting. It deals with the decision on the optimal size of the final tree.

Decision trees are the most susceptible algorithms to overfitting, as a tree that is too large risks overfitting the training data and poorly generalizing to new samples. On the other hand, a small tree might not capture important structural information about the sample space.

There are two types of pruning.

## Pre-pruning

Also called early-stopping, it consist on stopping the tree before it has completed classifying the training set. At each point in which the tree is split, the error is checked. If it does not decrease significantly, then the algorithm is stopped. It can produce underfitting.

## Post-pruning

It consists on pruning the tree after it has finished, that is, cutting back the tree. After the tree has been built, and in absence of pre-pruning, it may be overfitted, as the final leaves can consist of only one or a few data points: the tree has learned the data exactly, and may fail to predict new data.

# References

[Decision Tree Learning - Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning){: target="_blank"}

[Decision Tree from Scratch - Sebastian Mantey](https://github.com/SebastianMantey/Decision-Tree-from-Scratch){: target="_blank"}

[Shannon Entropy, Information Gain, and Picking Balls from Buckets
](https://medium.com/udacity/shannon-entropy-information-gain-and-picking-balls-from-buckets-5810d35d54b4){: target="_blank"}

[Plot the decision surface of a decision tree on the iris dataset](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html){: target="_blank"}

[Machine Learning: Pruning Decision Trees](https://www.displayr.com/machine-learning-pruning-decision-trees/){: target="_blank"}
