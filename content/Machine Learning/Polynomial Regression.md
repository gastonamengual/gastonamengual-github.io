Title: Polynomial Regression
Date: 2021.04.06
Summary: Definition. Application. Train and validation error for different degrees. Predictions for different degrees.

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from tqdm.notebook import tqdm
```


# Definition

Polynomial Regression is a form of regression analysis in which the relationship between the independent and the dependent variable is modeled as a $n$th degree polynomial.

It used when the relationship between the independent and dependent variable is not linear, and therefore includes non-liner terms. Mathematically, the model is

$$y = \beta_0 + \beta_1x + \beta_2 x^2 + \cdots + \beta_n x^n + \epsilon$$

The linear regression is a polynomial regression with degree 1.

Polynomial regression is related to the problems of underfitting and overfitting. For certain models, a polynomial function with degree 1 is not sufficient to fit the training samples (underfitting), while a polynomial of degree 4 approximates the true function almost perfectly. However, for higher degrees, the model will overfit the training data, i.e. it learns the noise of the training data. Underfitting and overfitting can be checked by calculating the MSE on the test set: the higher it is, the less likely the model generalizes correctly from the training data.

# Application


```python
def generate_points(n):
    np.random.seed(0)
    X = np.sort(np.random.rand(n))
    y = np.cos(1.5 * np.pi * X)
    return X.reshape(-1, 1), y
```


```python
X_train, y_train = generate_points(50)
y_train += np.random.randn(50) * 0.1
X_test = np.linspace(0, 1, 1000)
y_test = np.cos(1.5 * np.pi * X_test)
X_test = X_test.reshape(-1, 1)
```

### Train and validation error for different degrees


```python
num_degrees = 30
degrees = np.arange(1, num_degrees+1)
rmse_test = []
rmse_train = []

for degree in tqdm(degrees):

    polynomial_features = PolynomialFeatures(degree=degree)
    X_train_polynomial = polynomial_features.fit_transform(X_train)
    X_test_polynomial = polynomial_features.fit_transform(X_test)
    linear_regression = LinearRegression().fit(X_train_polynomial, y_train)
    
    # Train Error
    y_estimated = linear_regression.predict(X_train_polynomial)
    rmse = mse(y_train, y_estimated, squared=False)
    rmse_train.append(rmse)
    
    # Test Error
    y_estimated = linear_regression.predict(X_test_polynomial)
    rmse = mse(y_test, y_estimated, squared=False)
    rmse_test.append(rmse)
```


```python
fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)

axes[0].plot(degrees, rmse_train)
axes[1].plot(degrees, rmse_test)
```


    
![image alt text]({static}../images/polynomial_regression_1.png)
    



```python
optimum_degree = np.argmin(rmse_test)
print(f'The degree with the minimum test error is {optimum_degree}')
```

    The degree with the minimum test error is 6
    

### Predictions for different degrees


```python
degrees = [1, optimum_degree, 12, 17]

fig, axes = plt.subplots(2, 2, figsize=(16, 6))
for ax, degree in zip(axes.flatten(), degrees):

    polynomial_features = PolynomialFeatures(degree=degree)
    X_train_polynomial = polynomial_features.fit_transform(X_train)
    linear_regression = LinearRegression().fit(X_train_polynomial, y_train)
    X_test_polynomial = polynomial_features.fit_transform(X_test)
    y_estimated = linear_regression.predict(X_test_polynomial)
    
    rmse = mse(y_test, y_estimated, squared=False)
    
    ax.plot(X_test, y_estimated)
    ax.plot(X_test, y_test)
    ax.scatter(X_train, y_train)
    
```


    
![image alt text]({static}../images/polynomial_regression_2.png)


It can be observed that a 1st degree polynomial underfits the data, it is not complex enough to accurately capture relationships between the independent and dependent variables. A 6th polynomial fits the data the best, as it minimizes the RMSE on the test set. Adding more degrees causes the model to start overerfitting, as it becomes too adjusted to the training set, and fails to generalize to new unseen data.

# References

https://www.wikiwand.com/en/Polynomial_regression

https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py
