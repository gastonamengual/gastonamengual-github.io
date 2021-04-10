Title: Train-Test Split Evaluation
Date: 2021.01.10
Summary. Train-test split procedure. Cross-validation: K-folds, leave-p-out, leave-one-out.

```python
import numpy as np
```

# Train-Test Split General Procedure

The train-test split is a technique for evaluating the performance of a machine learning algorithm, used for supervised learning algorithms. 

Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. Train-test split is then used to:

* Estimate how accurately a predictive model will perform in practice and will generalize to an independent dataset (i.e., an unknown dataset). 
* Flag problems like overfitting or selection bias.

The procedure involves taking a data set and dividing it into two subsets:

* **Train Set**: it is used to fit the model. It represents the known data.
* **Test/Validation Set**: it is used to evaluate the model. The model makes predictions about the Test Set, and they are compared to the expected true values. It represents the unknown data.

The train-test split procedure is appropriate when the data set available is sufficiently large (in relation to the predictive modeling problem), meaning that there is enough data to split the data set into a train and test set that are suitable representations of the problem domain, covering all common cases and most uncommon cases in the domain. To ensure that both data sets are a representative sample of the original data set, the data points are assigned randomly to each set. For example, for classification problems that do not have a balanced number of examples for each class label, a stratified train-test split is used, which divides the data into train and test sets in a way that preserves the same proportions of examples in each class as observed in the original data.

The procedure has one main configuration parameter: the size of either the train or the test set. It is usually expressed as the percentage for the set. Although there is no optimal split percentage, the following configuration are common:

* Train: 80%, Test: 20%
* Train: 67%, Test: 33%
* Train: 50%, Test: 50%


```python
def train_test_split(X, y, train_proportion=0.8, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    data = np.column_stack((X, y))
    
    # Shuffle the data randomly
    shuffled_data = np.random.permutation(data)

    # Set the split point
    num_observations = data.shape[0]
    split_point = int(num_observations * train_proportion)

    # Train set
    train_set = shuffled_data[:split_point]
    X_train, y_train = train_set[:, :-1], train_set[:, -1]

    # Test set
    test_set = shuffled_data[split_point:]
    X_test, y_test = test_set[:, :-1], test_set[:, -1]
    
    return X_train, X_test, y_train, y_test
```


```python
X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]])
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(f'X Train set: {X_train}')
print(f'X Test set: {X_test}')
print(f'y Train set: {y_train}')
print(f'y Test set: {y_test}')
```

    X Train set: [[10 11]
     [ 2  3]
     [16 17]
     [ 0  1]
     [ 4  5]
     [18 19]
     [ 6  7]]
    X Test set: [[ 8  9]
     [12 13]
     [14 15]]
    y Train set: [5 1 8 0 2 9 3]
    y Test set: [4 6 7]
    

# Cross-validation

Cross-validation is a model evaluation procedure that allows to build confidence intervals for the evaluation metrics of an algorithm. It is applied to the train set after performing the train-validation split.

The procedure works as follows. The train set is split into $n$ different pairs of sets: a subtrain set and a subtest set. The model is then applied to every pair of sets, i.e. is trained with the subtrain set, and tested with the subtest set. By doing this, all of the points of the train set are part of both the subtrain and the subtest sets, allowing the algorithm to make predictions on all of the data. Then, for each pair a chosen evaluation metric is calculated. At the end of the procedure, $n$ values for the metric were calculating, which allows the construction of a confidence interval of the metrics, instead of one single value as in the traditional train-test split. 

Cross-validation is also used for hyperparameter tuning, i.e. estimate the optimal parameter for an algorithm, such as the number of nearest neighbors in $k$-NN, the optimal number of trees in Random Forests, and the type of kernel in an SVM. After performing cross-validation for several values of the hyperparameter, the one with the best confidence interval is chosen as the optimal. Afterwards, the algorithm is applied on the validation set with the optimal parameter, to get an estimate on the performance of the algorithm when it sees new data.

## $k$-Fold

In this approach, the training set is split into $k$ smaller sets or folds. For each of the $k$ folds, the following procedure is followed:

* Train the model using $k - 1$ of the folds as training data.

* Validate the resulting model on the remaining part of the data.

The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data.

The data should be previously shuffled. These methods do not compute all ways of splitting the original sample.

![image alt text]({static}../images/cross_validation_1.png)

The two following figures show a K-Fold Cross-validation for the number of trees in a Random Forest, and the $k$ parameter in $k$-NN, respectively. It can be seen that, for each value of the parameter, a boxplot can be constructed, as several (in this case, 7) metrics are calculated, one for each fold, giving a wider estimation on the behavior of the algorithm with unseen data.

![image alt text]({static}../images/random_forests_3.png)

![image alt text]({static}../images/k_nearest_neighbors_2.png)   


```python
def k_folds_generator(X, folds):
    
    cut_indexes = np.ceil(np.linspace(0, X_train.shape[0], folds + 1)).astype(int)

    for i in range(folds):

        train_1 = np.arange(0, cut_indexes[i])
        train_2 = np.arange(cut_indexes[i+1], X.shape[0])
        train_index = np.append(train_1, train_2)
        
        test_index = np.arange(cut_indexes[i], cut_indexes[i+1])
        
        yield train_index, test_index
```


```python
X_train = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100], [110, 120]])
y_train = np.array([10, 20, 30, 40, 50, 60, 70])

for train_index, test_index in k_folds_generator(X_train, folds=3):
    X_train_folds, X_test_folds = X_train[train_index], X_train[test_index]
    y_train_folds, y_test_folds = y_train[train_index], y_train[test_index]
    print(f'Train set index: {train_index} - Test set index: {test_index}')
```

    Train set index: [2 3 4 5] - Test set index: [0 1]
    Train set index: [0 1 4 5] - Test set index: [2 3]
    Train set index: [0 1 2 3] - Test set index: [4 5]
    

## Leave-*p*-Out

Leave-*p*-Out involves using $p$ observations as the test set, and the remaining observations as the training set, repeated on all the possible ways to divide the original sample into a training and validation set. 

This method require training and testing the model $C_{n}^{p}$ (binomial coefficient) times, where $n$ is the number of observations of the data set, which can be highly computationally expensive. For example, for $C_{40}^{10}$, that is, 40 observations leaving 10 out, it would take 847,660,528 sets.


```python
from sklearn.model_selection import LeavePOut
from scipy.special import comb

X_train = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100], [110, 120]])
y_train = np.array([10, 20, 30, 40, 50, 60, 70])

p = 2

print(f'Number of sets: {comb(X_train.shape[0], p):.2g}')

for i, (train_index, test_index) in enumerate(LeavePOut(p).split(X_train)):
    print(f'Train set index: {train_index} - Test set index: {test_index}')
```

    Number of sets: 15
    Train set index: [2 3 4 5] - Test set index: [0 1]
    Train set index: [1 3 4 5] - Test set index: [0 2]
    Train set index: [1 2 4 5] - Test set index: [0 3]
    Train set index: [1 2 3 5] - Test set index: [0 4]
    Train set index: [1 2 3 4] - Test set index: [0 5]
    Train set index: [0 3 4 5] - Test set index: [1 2]
    Train set index: [0 2 4 5] - Test set index: [1 3]
    Train set index: [0 2 3 5] - Test set index: [1 4]
    Train set index: [0 2 3 4] - Test set index: [1 5]
    Train set index: [0 1 4 5] - Test set index: [2 3]
    Train set index: [0 1 3 5] - Test set index: [2 4]
    Train set index: [0 1 3 4] - Test set index: [2 5]
    Train set index: [0 1 2 5] - Test set index: [3 4]
    Train set index: [0 1 2 4] - Test set index: [3 5]
    Train set index: [0 1 2 3] - Test set index: [4 5]
    

## Leave-One-Out

Leave-One-Out is equivalent to K-Fold with $n$ folds, where $n$ is the number of samples. Although it requieres less computation time than leave-*p*-Out, as there are only $C_{n}^{1}$ sets, rather than $C_{n}^{p}$, it may still take a large computation time.


```python
X_train = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100], [110, 120]])
y_train = np.array([10, 20, 30, 40, 50, 60, 70])
n = X_train.shape[0]

for train_index, test_index in k_folds_generator(X_train, folds=n):
    X_train_folds, X_test_folds = X_train[train_index], X_train[test_index]
    y_train_folds, y_test_folds = y_train[train_index], y_train[test_index]
    print(f'Train set index: {train_index} - Test set index: {test_index}')
```

    Train set index: [1 2 3 4 5] - Test set index: [0]
    Train set index: [0 2 3 4 5] - Test set index: [1]
    Train set index: [0 1 3 4 5] - Test set index: [2]
    Train set index: [0 1 2 4 5] - Test set index: [3]
    Train set index: [0 1 2 3 5] - Test set index: [4]
    Train set index: [0 1 2 3 4] - Test set index: [5]
    

# References

[Train-Test Split for Evaluating Machine Learning Algorithms](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/){: target="_blank"}

[Cross-validation - Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics)){: target="_blank"}

[scikit-learn - 3.1. Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html){: target="_blank"}
