Title: Random Forests
Date: 2021.04.01
Summary: Definition of Random Forest. Implementation from scratch. Application on classification and regression tasks. Choosing of optimum number of trees. 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mae
from sklearn.metrics import accuracy_score

from tqdm.notebook import tqdm
```


# Definition

Random forests are an ensemble learning method for classification and regression. They construct a multitude of decision trees at training time and return the class that is the mode of the classes (classification) or the average prediction (regression) of the individual trees. Random forests solve the overfitting problem of the decision trees, and apply the general technique of bootstrap aggregating. The decision trees are grown deep and the trees are not pruned, and will therefore have both high variance and low bias.

Combining predictions from multiple models in ensembles works better if the predictions from the sub-models are not or at least weakly correlated. In the data, there can be certain features that are very strong predictors for the target output, and will be selected in many trees, causing them to become correlated. To ensure that all trees are uncorrelated, random forests use a modified decision tree algorithm: instead of using all the features, it selects, at each candidate split in the learning process, a random subset of the features. Common choices of number of features are $\sqrt{\text{num features}}$ and $log_2(\text{num features})$.

The only parameters when bagging decision trees is the number of trees to include. This can be chosen by increasing the number of trees on run after run until the accuracy begins to stop showing improvement (e.g. on a cross validation test harness). Very large numbers of models may take a long time to prepare, but will not cause the training data to overfit.

The major disadvantage of random forests is their complexity. They required much more computational resources, due to the large number of decision trees joined together, and they require much more time to train than other comparable algorithms.

# Implementation


```python
def random_forest_train(X, y, num_trees, task):
    
    num_samples = X.shape[0]
    
    random_forest = []

    for _ in range(num_trees):
        indexes = np.random.choice(num_samples, size=num_samples, replace=True)
        X_sample, y_sample = X[indexes], y[indexes]

        if task == 'classification':
            tree = DecisionTreeClassifier(max_features='sqrt')
        if task == 'regression':
            tree = DecisionTreeRegressor(max_features='sqrt')
            
        tree.fit(X_sample, y_sample)
        random_forest.append(tree)
    
    return random_forest

def random_forest_predict(random_forest, X, task):
    single_trees_predictions = np.array([tree.predict(X) for tree in random_forest]).T
    
    if task == 'classification':
        predictions, _ = mode(single_trees_predictions, axis=1)
    if task == 'regression':
        predictions = single_trees_predictions.mean(axis=1).flatten()
    
    return predictions
```

# Classification Application


```python
digits = datasets.load_digits()
X = digits.data
y = digits.target
train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size=0.2)
```


```python
num_replicates = 1000
```

## Optimum Number of Trees

To find the optimum number of trees, K-Folds Cross-validation will be used to calculate the random forest accuracy for each number of trees. 


```python
folds = 7
kf = KFold(n_splits=folds)

num_trees_values = np.arange(1, 51)
num_trees_accuracy = []

for n in tqdm(num_trees_values):
    
    folds_accuracy = []
    
    for train_index, test_index in kf.split(train_set_X):
        X_train_fold, X_test_fold = train_set_X[train_index], train_set_X[test_index]
        y_train_fold, y_test_fold = train_set_y[train_index], train_set_y[test_index]

        random_forest = random_forest_train(X_train_fold, y_train_fold, num_trees=n, task='classification')
        predictions = random_forest_predict(random_forest, X_test_fold, task='classification')
        
        accuracy = accuracy_score(y_test_fold, predictions)
        folds_accuracy.append(accuracy)
        
    num_trees_accuracy.append(folds_accuracy)
```




```python
accuracy_training_df = pd.DataFrame(data=num_trees_accuracy, index=num_trees_values)

best_n_training = np.argmax(accuracy_training_df.median(axis=1)) + 1
best_accuracy_training = np.max(accuracy_training_df.median(axis=1))

ax = sns.boxplot(data=accuracy_training_df.T)
```

    The value of n that presented the best accuracy median in 7 folds is: 35 with value of 0.976
    


    
![image alt text]({static}../images/random_forests_1.png)
    


## Predictions: Single Decision Trees vs Random Forest

The accuracy for 1000 random forest will be calculated, predicting the values of the Test Set using the optimum number of trees value found.


```python
accuracies_decision_trees = []
accuracies_random_forests = []

for _ in tqdm(range(num_replicates)):
    
    # Single Decision Tree
    decision_tree = DecisionTreeClassifier().fit(train_set_X, train_set_y)
    predictions = decision_tree.predict(test_set_X) 
    accuracy = accuracy_score(test_set_y, predictions)
    accuracies_decision_trees.append(accuracy)
    
    # Random Forest
    random_forest = random_forest_train(train_set_X, train_set_y, num_trees=best_n_training, task='classification')
    predictions = random_forest_predict(random_forest, test_set_X, task='classification')
    accuracy = accuracy_score(test_set_y, predictions)
    accuracies_random_forests.append(accuracy)
```

    
    


```python
fig, axes = plt.subplots(1, 2)
axes[0].hist(accuracies_decision_trees)
axes[1].hist(accuracies_random_forests)
```


    
![image alt text]({static}../images/random_forests_2.png)
    



```python
overlapping = np.mean(accuracies_decision_trees == accuracies_random_forests)
print(f'Histograms overlapping = {overlapping:.0f}%')
```

    Histograms overlapping = 0%
    

# Regression Application


```python
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size=0.2)
```


```python
num_replicates = 1000
```

## Optimum Number of Trees

To find the optimum number of trees, K-Folds Cross-validation will be used to calculate the random forest RMSE for each number of trees.


```python
folds = 7
kf = KFold(n_splits=folds)

num_trees_values = np.arange(1, 51)
num_trees_rmse = []

for n in tqdm(num_trees_values):
    
    folds_rmse = []
    
    for train_index, test_index in kf.split(train_set_X):
        X_train_fold, X_test_fold = train_set_X[train_index], train_set_X[test_index]
        y_train_fold, y_test_fold = train_set_y[train_index], train_set_y[test_index]

        random_forest = random_forest_train(X_train_fold, y_train_fold, num_trees=n, task='regression')
        predictions = random_forest_predict(random_forest, X_test_fold, task='regression')
        
        rmse = mae(y_test_fold, predictions, squared=False)
        folds_rmse.append(rmse)
        
    num_trees_rmse.append(folds_rmse)
```



```python
rmse_training_df = pd.DataFrame(data=num_trees_rmse, index=num_trees_values)

best_n_training = np.argmin(rmse_training_df.median(axis=1)) + 1
best_accuracy_training = np.min(rmse_training_df.median(axis=1))

ax = sns.boxplot(data=rmse_training_df.T)
```

    The value of n that presented the best accuracy median in 7 folds is: 38 with value of 54.9
    


    
![image alt text]({static}../images/random_forests_3.png)
    


## Predictions: Single Decision Trees vs Random Forest

The RMSE for 1000 random forest will be calculated, predicting the values of the Test Set using the optimum number of trees value found.


```python
rmses_decision_trees = []
rmses_random_forests = []

for _ in tqdm(range(num_replicates)):
    
    # Single Decision Tree
    predictions = DecisionTreeRegressor().fit(train_set_X, train_set_y).predict(test_set_X) 
    rmse = mae(test_set_y, predictions, squared=False)
    rmses_decision_trees.append(rmse)
    
    # Random Forest
    random_forest = random_forest_train(train_set_X, train_set_y, num_trees=best_n_training, task='regression')
    predictions = random_forest_predict(random_forest, test_set_X, task='regression')
    rmse = mae(test_set_y, predictions, squared=False)
    rmses_random_forests.append(rmse)
```


    


```python
fig, axes = plt.subplots(1, 2)
axes[0].hist(rmses_decision_trees)
axes[1].hist(rmses_random_forests)
```


    
![image alt text]({static}../images/random_forests_4.png)
    



```python
overlapping = np.mean(rmses_random_forests == rmses_decision_trees)
print(f'Histograms overlapping = {overlapping:.0f}%')
```

    Histograms overlapping = 0%
    

# Conclusions

From the above histograms, it can be observed that the random forest improves the performance of the model significantly, and, as the histograms do not overlap, a random forest is always better than a single decision tree in those data sets. Therefore, it helps to solve the problem of overfitting of the decision trees. Moreover, from the boxplots, it can be concluded that, at a certain point, increasing the number of trees does not improve the performance of the Random Forest significantly.

# References

[Random Forest - Wikipedia](https://www.wikiwand.com/en/Random_forest){: target="_blank"}

[Bagging and Random Forest Ensemble Algorithms for Machine Learning](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/){: target="_blank"}
