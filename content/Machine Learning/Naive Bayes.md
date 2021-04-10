Title: Naive Bayes
Date: 2021.04.05
Summary: Definition. Implementation. Application.

```python
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


# Definition

Naive Bayes classifiers are a set of supervised learning algorithms based on Bayes' theorem. It is especially appropriate when the dimension of the feature space is high.

Naive Bayes' main assumption is that features are mutually independent. Although this is generally not true (therefore *naive*), it simplifies the estimation dramatically, as the individual class-conditional marginal densities can be estimated separately.

## Model

Let $x = (x_1, \cdots, x_n)$ be a feature vector, and $y$ be a class variable. Bayes' theorem states

$$P(y | x_1, \cdots, x_n)  = \dfrac{P(y) P(x | y)}{P(x_1, \cdots, x_n)} = \dfrac{\text{posterior} \cdot \text{likelihood}}{\text{prior}}$$

Then, due to the **naive conditional independence** assumption,

$$P(x_i | x_1, \cdots, x_{x-1}, x_{i+1}, \cdots, x_n, y) = P(x_i | y)$$

and the relationship can be simplified

$$P(y | x_1, \cdots, x_n)  = \dfrac{P(y) \prod_{i=1}^{n} P(x_i|y)}{P(x_1, \cdots, x_n)}$$

Since $P(x_1, \cdots, x_n)$ is constant given the input, the following classification rule is defined:

$$P(y | x_1, \cdots, x_n)  \propto P(y) \prod_{i=1}^{n} P(x_i|y)$$

$$\hat{y} = \arg \max_{y} P(y) \prod_{i=1}^{n} P(x_i|y)$$

To minimize computational cost, the logarithm is applied

$$\hat{y} = \arg \max_{y} \sum_{i=1}^{n} \log({P(x_i|y)}) + \log{P(y)}$$

To estimate $P(y)$ and $P(x_i|y)$ the Maximum a Posteriori can be used.

The prior distribution $P(y)$ is calculated as the relative frequency of class $y$ in the training set.

## Gaussian Naive Bayes

Usually, for continuous features, it is assumed that the continuous values associated with each class $y_j$ are distributed according to a normal distribution. Therefore, it is equal to the value of the normal probability density distribution at $x_i$:

$$P(x_i | y_j) = f(x_i, \mu_{y_j}, \sigma^2_{y_j}) = \frac{1}{\sqrt{2\pi\sigma^2_{y_j}}} \exp\left(-\frac{(x_i - \mu_{y_j})^2}{2\sigma^2_{y_j}}\right)$$

# Implementation


```python
def naive_bayes_fit(X, y):
    num_samples = X.shape[0]
    classes = np.unique(y)
    num_classes = classes.shape[0]
    
    prior_per_class =  []
    mean_per_class = []
    variance_per_class = []
    
    for class_ in classes:
        
        x_class = X[y==class_]
        
        prior_per_class.append(x_class.shape[0] / num_samples)
        mean_per_class.append(x_class.mean(axis=0))
        variance_per_class.append(x_class.var(axis=0))

    return classes, np.array(prior_per_class), np.array(mean_per_class), np.array(variance_per_class)
```


```python
def naive_bayes_predict(X, classes, prior_per_class, mean_per_class, variance_per_class):
    
    predictions = []
    
    for x in X:
        
        classes_posterior = []

        # Calculate posterior probability for each class
        for i in range(len(classes)):
            
            prior = np.log(prior_per_class[i])
            
            class_mean = mean_per_class[i]
            class_variance = variance_per_class[i]
            posterior = prior + np.sum(np.log(norm.pdf(x, class_mean, np.sqrt(class_variance)))) 
            
            classes_posterior.append(posterior)

        # Predict class with highest posterior probability
        predicted_class = classes[np.argmax(classes_posterior)]
        predictions.append(predicted_class)
    
    return np.array(predictions)
```

# Application


```python
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```

### Train


```python
classes, prior_per_class, mean_per_class, variance_per_class = naive_bayes_fit(X_train, y_train)
```

### Test


```python
predictions = naive_bayes_predict(X_test, classes, prior_per_class, mean_per_class, variance_per_class)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy = {accuracy:.3f}')
```

    Accuracy = 0.833
    

### Prediction Map


```python
x_min, x_max = X.T[0].min() - 1, X.T[0].max() + 1
y_min, y_max = X.T[1].min() - 1, X.T[1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

data = np.column_stack((xx.flatten(), yy.flatten()))
Z = naive_bayes_predict(data, classes, prior_per_class, mean_per_class, variance_per_class)
Z = Z.reshape(xx.shape)
```


```python
plt.contourf(xx, yy, Z)
plt.scatter(*X.T)
```


    
![image alt text]({static}../images/naive_bayes_1.png)
    


# References

[scikit-learn Naive Bayes](scikit-learn.org/stable/modules/naive_bayes.html){: target="_blank"}

[Gaussian Naive Bayes Classifier: Iris data set](xavierbourretsicotte.github.io/Naive_Bayes_Classifier.html){: target="_blank"}

[MLfromscratch](github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/naivebayes.py){: target="_blank"}

[Naive Bayes classifier - Wikipedia](en.wikipedia.org/wiki/Naive_Bayes_classifier){: target="_blank"}
