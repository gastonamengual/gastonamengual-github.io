Title: Feature Scaling
Date: 2021.03.25
Summary: Definition. Explanation and implementation of min-max normalization, mean normalization, and standardization. Application to data set.

```python
import numpy as np
from sklearn import datasets
from IPython.display import display, Markdown
```

# Definition

Feature scaling is a part of the data pre-processing stage. It is applied when the range of the values of the input data are very different in each feature. If the value of certain features is very high in comparison to others, this may produce, in some cases, that they have greater impact on the algorithm, which will give them more importance and weight, than other more important features that have lower magnitude. The algorithm then will fail to give precise estimations.

Some algorithms sensitive to feature scaling are Linear and Logistic Regression, *k*-Means, *k*-NN, PCA, and SVM. On the other hand, other algorithms like Decision Trees are insensitive to feature scaling.

There are two kinds of Feature Scaling, normalization and standardization. 

Let $x_i$ be an observation of a feature vector $x$, $\bar{x}$ the mean of $x$, and $x_{min}$ and $x_{max}$ the minimum and maximum value of $x$, respectively.

# Normalization

Normalization is a scaling technique in which values are shifted and rescaled to be between two numbers. Typically, the values are scaled to the range $[0,1]$ or $[-1,1]$. It is useful when the distribution of the data does not follow a Gaussian distribution and is sensible to outliers.

## Min-Max Normalization

It consists in scaling $x_i$ to the range $[a, b]$. The general formula is

$$x_i' = a + \dfrac{(x_i - x_{min})(b-a)}{x_{max} - x_{min}}$$

Intuitively, if $a=0$ and $b=1$, when $x_i$ is the minimum value, the numerator will be 0, and hence $x_i' = 0$; when $x_i$ is the maximum value, the numerator is equal to the denominator and thus the value of $x_i' = 1$. If $x_i$ is between the minimum and the maximum value, then the $x_i'$ is between $0$ and $1$.


```python
def min_max_normalization(X, a=0, b=1):
    scaled_data = a + (X - X.min(axis=0)) * (b - a) / (X.max(axis=0) - X.min(axis=0))
    return scaled_data
```

## Mean Normalization

Another form of normalization consists of subtracting the mean from the observation. The formula is

$$x' = \dfrac{x_i - \bar{x}}{x_{max} - x_{min}}$$

This method scales the data between -1 and 1.


```python
def mean_normalization(X):
    scaled_data = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return scaled_data
```

# Standardization

Standardization transforms the data to have a mean equal to $0$ (subtracting the mean in the numerator) and a variance equal to $1$, and it makes the data have no units. It is especially useful in cases where the data follows a Gaussian distribution. Also, unlike normalization, standardization does not have a bounding range, so, even if there are outliers in the data, they will have less impact on the standardization process. The formula is

$$x' = \dfrac{x_i - \bar{x}}{\sigma}$$

where $\sigma$ is the standard deviation of the feature observations.


```python
def standardization(X):
    scaled_data = (X - X.mean(axis=0)) / X.std(axis=0)
    return scaled_data
```

# Application

The Wine data set will be used to illustrate how feature scaling works. From this data sets, 6 features will be chosen, whose range of values varies significantly in scale.


```python
wine = datasets.load_wine()
X = wine.data[:, (0, 1, 3, 4, 7, 12)]
features = ['Alcohol', 'Malic Acid', 'Alcalinity Of Ash', 'Magnesium', 'Nonflavanoid Phenols', 'Proline']
```


```python
min_values, max_values = X.min(axis=0), X.max(axis=0)

display(Markdown('**Original Ranges**'))
for feature, min_value, max_value in zip(features, min_values, max_values):
    display(Markdown(f'{feature} = $[{min_value}, {max_value}]$'))
```


**Original Ranges**



Alcohol = $[11.03, 14.83]$



Malic Acid = $[0.74, 5.8]$



Alcalinity Of Ash = $[10.6, 30.0]$



Magnesium = $[70.0, 162.0]$



Nonflavanoid Phenols = $[0.13, 0.66]$



Proline = $[278.0, 1680.0]$



```python
scaled_data = min_max_normalization(X)
min_values, max_values = scaled_data.min(axis=0), scaled_data.max(axis=0)

display(Markdown('**Min-Max Normalization**'))
for feature, min_value, max_value in zip(features, min_values, max_values):
    display(Markdown(f'{feature} = $[{min_value}, {max_value}]$'))
```


**Min-Max Normalization**



Alcohol = $[0.0, 1.0]$



Malic Acid = $[0.0, 1.0]$



Alcalinity Of Ash = $[0.0, 1.0]$



Magnesium = $[0.0, 1.0]$



Nonflavanoid Phenols = $[0.0, 1.0]$



Proline = $[0.0, 1.0]$



```python
scaled_data = mean_normalization(X)
min_values, max_values = scaled_data.min(axis=0), scaled_data.max(axis=0)

display(Markdown('**Mean Normalization**'))
for feature, min_value, max_value in zip(features, min_values, max_values):
    display(Markdown(f'{feature} = $[{min_value:.2f}, {max_value:.2f}]$'))
```


**Mean Normalization**



Alcohol = $[-0.52, 0.48]$



Malic Acid = $[-0.32, 0.68]$



Alcalinity Of Ash = $[-0.46, 0.54]$



Magnesium = $[-0.32, 0.68]$



Nonflavanoid Phenols = $[-0.44, 0.56]$



Proline = $[-0.33, 0.67]$



```python
scaled_data = standardization(X)
min_values, max_values = scaled_data.min(axis=0), scaled_data.max(axis=0)

display(Markdown('**Standardization**'))
for feature, min_value, max_value in zip(features, min_values, max_values):
    display(Markdown(f'{feature} = $[{min_value:.2f}, {max_value:.2f}]$'))
```


**Standardization**



Alcohol = $[-2.43, 2.26]$



Malic Acid = $[-1.43, 3.11]$



Alcalinity Of Ash = $[-2.67, 3.15]$



Magnesium = $[-2.09, 4.37]$



Nonflavanoid Phenols = $[-1.87, 2.40]$



Proline = $[-1.49, 2.97]$

<br>

In conclusion, using one of the methods described, all the features values are on the same scale, and the scaled data can be used as input of the algorithms that require feature scaling as a pre-processing step.

# References

[Feature Scaling](https://en.wikipedia.org/wiki/Feature_scaling){: target="_blank"}

[Feature Scaling for Machine Learning](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/){: target="_blank"}
