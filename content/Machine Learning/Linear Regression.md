Title: Linear Regression
Date: 2021.01.20
Summary: Simple Linear Regression. Assumptions. Ordinary Least Squares. Implementation. Application and testing of assumptions. Confidence and prediction intervals. Multiple Linear Regression.

## Table of Contents

[Simple Linear Regression](#Simple-Linear-Regression)

$\quad$ [Assumptions](#Assumptions)

$\quad$ [Ordinary Least Squares Simple](#Ordinary-Least-Squares-Simple)

$\quad$ [SLR-Implementation](#SLR-Implementation)

$\quad$ [SLR-Application](#SLR-Application)

$\quad$ [Confidence and Prediction](#Confidence-and-Prediction)

<br>

[Multiple Linear Regression](#Multiple-Linear-Regression)

$\quad$ [Ordinary Least Squares Multiple](#Ordinary-Least-Squares-Multiple)

$\quad$ [MLS-Implementation](#MLS-Implementation)

$\quad$ [MLS-Application](#MLS-Application)


```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
```


# Simple Linear Regression

Linear regression is the simplest regression type. If the linear regression model has only one independent variable, then it is called Simple Linear Regression (SLR). When it has more than one independent variables, it is called Multiple Linear Regression. 

<br>

The **true SLR model** describes the true relationship between independent and dependent variables, and it is achieved by carrying out the regression on the whole population, either with complete data or theoretical calculations. It is described by:

$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$

where $x_i$ is the independent variable, $y_i$ is the true dependent variable, $(x_i, y_i)$ constitute a true data pair or point, $\beta_0$ and $\beta_1$ are the true parameters, and $\epsilon_i$ is the error term. The **errors** are the deviations of the dependent variable observations from an unobservable function that relates the independent variable to the dependent variable.

<br>

However, in most cases the attributes of the whole population cannot be measured, so a sample is taken instead, and an **estimated SLR model** is constructed instead:

$$\hat{y_i} = \hat{\beta_0} + \hat{\beta_1} x_i + r_i$$

where $x_i$ is the independent variable, $\hat{y_i}$ is the estimated dependent variable, $(x_i, \hat{y_i})$ constitute a estimated data pair or point, $\hat{\beta_0}$ and $\hat{\beta_1}$ are the estimated parameters, and $r_i$ is the residual term. The **residuals** are the deviations of the dependent variable observations from the fitted function. The residuals are defined as:

$$\hat{r_i} = \hat{y_i} - \hat{\beta_0} - \hat{\beta_1} x_i$$

Finally, the minimization problem, the **sum of squared residuals**, is:

$$\min\limits_{\hat{\beta_0}, \hat{\beta_1}} Q(\hat{\beta_0}, \hat{\beta_1}) = \sum_{i=1}^{n}{\hat{r}_i^2 = \sum_{i=1}^{n}(\hat{y_i} - \hat{\beta_0} -\hat{\beta_1} x_i)^2}$$

As can be observed, the SLR model is a line, and the objective is to find the slope and intercept, i.e. the values of $\hat{\beta_0}$ and $\hat{\beta_1}$ that best fits the data points to the line, minimizing the difference of the true observed dependent variables and the estimated dependent variables. 

## Assumptions

To apply the Simple Linear Regression model (and more generally the Linear Regression model) to a data set, it must meet a set of assumptions. If they are not met, then the results of the model can be mistaken or not precise.  

### Linearity

There is a linear relationship between the independent variable $x$ and the dependent variable $y$, that is, the expected value of the dependent variable is a straight-line function of the independent variable. 

<br>

**How to test?**

Linearity is usually evident in a plot of the independent variable versus the dependent variable (1), or in a plot of residuals versus the independent variable. In (1) the points should be symmetrically distributed around a diagonal line, while in (2) they should be distributed around a horizontal line, with a roughly constant variance. The latter approach is better, as it removes the visual distraction of a sloping pattern.

<br>

**What if the model is nonlinear?**

If the model is nonlinear, the predictions made are likely to be mistaken, especially when extrapolating beyond the range of the sample data. Several transformations can be applied nonlinear data (see [here](https://people.revoledu.com/kardi/tutorial/Regression/nonlinear/NonLinearTransformation.htm) for more details), or another independent variable can be added.

### Independence

The residuals must be independent.

<br>
**How to test?**

Independence can be tested with an autocorrelation plot of the residuals. Ideally, most of the residuals should fall within the 95% confidence bands around zero, where $n$ is the sample size. For a more formal and precise test, the [Durbin-Watson Test](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic) can be conducted.

<br>
**What if the residuals are not independent?**

Violations of independence are potentially very serious in time series regression models. To fix the issue, there are several options. For positive serial correlation, add lags of the dependent and/or independent variable to the model. For negative serial correlation, check to make sure that none of the variables are overdifferenced. For seasonal correlation, add seasonal dummy variables to the model.

### Homoscedasticity

The residuals must have constant variance at every level of the independent variable.

<br>
**How to test?**

Formal tests of heteroscedasticity can be studied [here](https://medium.com/@remycanario17/tests-for-heteroskedasticity-in-python-208a0fdb04ab). However, for a simpler way to estimate whether the residuals are heteroscedastic, apply resampling to a partition of continuous values of the residuals, and then construct confidence intervals for each of them. If all such CIs overlap with each other, then it is probable that the residuals are not heteroscedastic.

<br>
**What if the residuals are not independent?**

Transform the dependent variable, e.g., the log of the dependent variable, redefine the dependent variable, e.g., use a rate rather than the raw value, or use weighted regression.

### Normality

The residuals must be normally distributed with mean $0$.

<br>
**How to test?**

The normality can be checked using statistical tests like Shapiro-Wilk, Kolmogorov-Smironov, Jarque-Barre, or D’Agostino-Pearson. However, they are sensitive to large sample sizes, as they often conclude that the residuals are not normal when the sample size is large. The value of the mean can be tested with the T-test for the mean of ONE group of scores.

<br>
**What if the residuals are not independent?**

Violations in normality create problems for determining whether model coefficients are significantly different from zero and for calculating confidence intervals for forecasts. Sometimes the residuals distribution is "skewed" by the presence of a few large outliers. To correct this problem, verify that outliers do not have great impact on the distribution, and apply a nonlinear transformation to the independent and/or dependent variable.

## Ordinary Least Squares Simple

Ordinary Least Squares (OLS) is a type of linear least squares method for estimating the unknown parameters in a linear regression model.

The sum of squared residuals function is defined as

$$Q(\hat{\beta_0}, \hat{\beta_1}) = \sum_{i=1}^{n}(y_{i}-\hat{\beta_0} -\hat{\beta_1} x_{i})^2$$

The values of $\hat{\beta_0}$ and $\hat{\beta_1}$ that minimizes the function are found as follows:

**1. Take the partial derivatives with respect to $\hat{\beta_0}$ and $\hat{\beta_1}$.**

$\frac{\partial }{\partial \hat{\beta_0}}\sum_{i=1}^{n}(\hat{y_i}-\hat{\beta_0} -\hat{\beta_1} x_i)^2 = -2 \sum_{i=1}^{n}(\hat{y_i}-\hat{\beta_0} -\hat{\beta_1} x_i)$

$\frac{\partial }{\partial \hat{\beta_1}}\sum_{i=1}^{n}(\hat{y_i}-\hat{\beta_0} -\hat{\beta_1} x_i)^2 = -2 \sum_{i=1}^{n} x_i(\hat{y_i}-\hat{\beta_0} -\hat{\beta_1} x_i)$

<br>

**2. Set the partial derivatives equal to 0.**

$\sum_{i=1}^{n}(\hat{y_i}-\hat{\beta_0} -\hat{\beta_1} x_i) = 0 \quad (1)$

$\sum_{i=1}^{n} x_i(\hat{y_i}-\hat{\beta_0} -\hat{\beta_1} x_i) = 0 \quad (2)$

(Note that $-2$ can be simplified)

<br>

**3. Solve for  $\hat{\beta_0}$ and $\hat{\beta_1}$.**

*Solving for $\hat{\beta_0}$ in $(1)$,*

$\sum_{i=1}^{n}(\hat{y_i}-\hat{\beta_0} -\hat{\beta_1} x_i) = 0$

$\sum_{i=1}^{n} \hat{y_i} - \sum_{i=1}^{n} \hat{\beta_0} - \sum_{i=1}^{n} \hat{\beta_1} x_i = 0$

$\sum_{i=1}^{n} \hat{y_i} - n \hat{\beta_0} - \hat{\beta_1} \sum_{i=1}^{n} x_i = 0$

$\hat{\beta_0} = \dfrac{\sum_{i=1}^{n} \hat{y_i}}{n} - \dfrac{\hat{\beta_1} \sum_{i=1}^{n} x_i}{n}$

$$\hat{\beta_0} = \bar{y} - \hat{\beta_1} \bar{x}$$

*Replacing $\hat{\beta_0}$ in $(2)$ and solving for $\hat{\beta_1}$,*

$\sum_{i=1}^{n} x_i(\hat{y_i}- \bar{y} + \hat{\beta_1}( \bar{x} - x_i)) = 0$

$\sum_{i=1}^{n} x_i(\hat{y_i}- \bar{y}) = \hat{\beta_1} \sum_{i=1}^{n} x_i (\bar{x} - x_i)$

$\hat{\beta_1} = \dfrac{\sum_{i=1}^{n} x_i (\hat{y_i} - \bar{y})}{\sum_{i=1}^{n} x_i (x_i - \bar{x})}$

It can be further demonstrated that the expression above equals

$$\hat{\beta_1} = \dfrac{\sum_{i=1}^{n} (x_i - \bar{x}) (\hat{y_i} - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$


```python
def ordinary_least_squares_simple(x, y):
    x_ = x.flatten()
    beta_1 = np.sum((x_ - x_.mean()) * (y - y.mean())) / ((x_ - x_.mean())**2).sum()
    beta_0 = y.mean() - beta_1 * x_.mean()
    
    return beta_0, beta_1
```

## SLR Implementation


```python
def simple_liner_regression_fit(x, y):
    beta_0, beta_1 = ordinary_least_squares_simple(x, y)
    return beta_0, beta_1
```


```python
def simple_linear_regression_predict(x, beta_0, beta_1):
    predictions = beta_0 + x * beta_1
    return predictions
```

## SLR Application

**Data set**


```python
x, y = make_regression(n_samples=5000, n_features=1, noise=20, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

**Train**


```python
beta_0, beta_1 = simple_liner_regression_fit(x_train, y_train)

print(f'β0: {beta_0:.4f}')
print(f'β1: {beta_1:.4f}')
```

    β0: -0.3420
    β1: 10.2998
    

**Test**


```python
y_estimated = simple_linear_regression_predict(x_test, beta_0, beta_1)
rmse = mse(y_test, y_estimated, squared=False)
print(f'RMSE: {rmse:.4f}')
```

    RMSE: 20.4741
    

**Line Prediction**


```python
beta_0, beta_1 = simple_liner_regression_fit(x, y)
y_estimated = simple_linear_regression_predict(x, beta_0, beta_1)
```


```python
plt.scatter(x, y)
plt.plot(x, y_estimated)
```


    
![image alt text]({static}../images/linear_regression_1.png)
    


The SLR model is given then by the line given by the function

$$\hat{y_i} = -0.21 + 10.77 \; x_i + r_i$$

To know for certain that this model is accurate and precise, the four assumptions will be verified.


```python
residuals = y - y_estimated.flatten()
```
<br>

**Assumption 1: Linearity**


```python
fig, axes = plt.subplots(1, 2)

axes[0].scatter(x, y)
axes[1].scatter(x, residuals)
```


![image alt text]({static}../images/linear_regression_2.png)
    


A clear linear relationship can be observed between $x$ and $y$, while most of the residuals are distributed around a horizontal line, with a roughly constant variance. 

<br>

**Assumption 2: Independence of errors**


```python
pd.plotting.autocorrelation_plot(residuals)
```
    


    
![image alt text]({static}../images/linear_regression_3.png)    


It can be seen that the autocorrelation falls between the confidence interval, and I don't know what that means, and what CI we're talking about...

<br>

**Assumption 3: Homoscedasticity**


```python
n = 3
sets = np.linspace(0, residuals.size, n + 1).astype(int)

for i in range(n):
    
    variances = []
    for _ in range(10000):
        resample = np.random.choice(residuals[sets[i]:sets[i+1]], sets[1])
        variances.append(resample.var())
    
    print(f'CI for Sample {i+1}: {np.percentile(variances, 2.5):.2f} - {np.percentile(variances, 97.5):.2f}')
    plt.hist(variances)
    
```

    CI for Sample 1: 382.43 - 438.31
    CI for Sample 2: 377.82 - 436.26
    CI for Sample 3: 365.82 - 418.22
    


    
![image alt text]({static}../images/linear_regression_4.png)
    


It can be seen the the three histograms overlap in their CI, show that there is a high probability that the residuals are not heteroscedastic. For the purposes of this work, this conclusion is enough to consider the residuals homoscedastic, while formal testing is encouraged.

<br>

**Assumption 4: Normality**


```python
_, pvalue = stats.ttest_1samp(residuals, popmean=0)

if pvalue < 0.05:
    print('Null Hypothesis: "The CI for the residual population mean contains 0" rejected.')
else:
    print('Null Hypothesis: "The CI for the residual population mean contains 0" not rejected.')

_, pvalue = stats.normaltest(residuals)

if pvalue < 0.05:
    print('Null Hypothesis: "the residuals sample comes from a normal distribution" rejected.')
else:
    print('Null Hypothesis: "the residuals sample comes from a normal distribution" not rejected.')
```

Null Hypothesis: "The CI for the residual population mean contains 0" not rejected.

Null Hypothesis: "the residuals sample comes from a normal distribution" not rejected.
    
<br>

Both Null Hypothesis were not rejected, and it can be concluded that the residuals follow a normal distribution with mean $0$.

<br>

**In conclusion, the data set meets the four SLR assumptions.**

## Confidence and Prediction

Suppose a new value of $x$, $x_{new}$. There are two questions that can be asked:

* Which is the most probable value (mean response) for a new observation $x_{new}$?

* What will be the range of possible response values for $y_{new}$ for a new observation $x_{new}$?

$\hat{y} = \beta_0 + \beta_1 x_{new}$ is the best answer to the first question, whereas to answer the second question, a confidence interval for $y_{new}$ must be built.

### Confidence interval for mean response 

It is also called a *t*-interval. The general formula is

$$\text{Sample estimate} \pm (\text{t-multiplier} \times \text{standard error})$$

Mathematically, 

$$\hat{y}_h \pm t_{(\alpha/2, n-2)} \times \sqrt{MSE \times \left( \frac{1}{n} + \dfrac{(x_{new}-\bar{x})^2}{\sum(x_i-\bar{x})^2}\right)}$$

where:

* $\hat{y}_h$ is the fitted or predicted value of the response when the predictor is $x_h$. 

* $t_{(\alpha/2, n-2)}$ is the "t-multiplier." (with n-2 degrees of freedom).
 
* $\sqrt{MSE \times \left( \frac{1}{n} + \frac{(x_h-\bar{x})^2}{\sum(x_i-\bar{x})^2}\right)}$ is the "standard error of the fit," which depends on the mean square error (MSE), the sample size (n), how far in squared units the predictor value $x_h$ is from the average of the predictor values $\bar{x}$, or $(x_h-\bar{x})^2$, and the sum of the squared distances of the predictor values $x_i$ from the average of the predictor values $\bar{x}$, or $\sum(x_i-\bar{x})^2$.

From the formula above, it can be concluded that:

* As the MSE decreases, the width of the interval decreases.
* As the sample size increases, the width of the interval decreases. 
* The more spread out the predictor values, the narrower the interval.
* The closer $x_h$ is to the average of the sample's predictor values, the narrower the interval.

### Prediction interval for $y_{new}$

It is similar to the formula explained above, except that an extra MSE term is added:

$$\hat{y}_h \pm t_{(\alpha/2, n-2)} \times \sqrt{MSE \times \left(1+ \frac{1}{n} + \dfrac{(x_{new}-\bar{x})^2}{\sum(x_i-\bar{x})^2}\right)}$$

It must be noticed that, because of the extra MSE term, a confidence interval for $y_{new}$ at $x_h$ will always be wider than the corresponding confidence interval for $\mu_y$ at $x_h$.


```python
def mean_response_interval(x_new, x, residuals):
    
    # 95% Confidence: 0.025
    t_multiplier = stats.t(x.shape[0] - 2).ppf(0.975)
    
    mse = np.sum(residuals**2) / (x.shape[0] - 2)
    
    x_size = x.shape[0]
    x_mean = x.mean()

    mean_margin = t_multiplier * np.sqrt(mse * (1 / x_size + (x_new - x_mean)**2 / (np.sum((x - x_mean)**2) / (x_size - 1))))
    
    return mean_margin

def prediction_interval(x_new, x, residuals):
    
    # 95% Confidence: 0.025
    t_multiplier = stats.t(x.shape[0] - 2).ppf(0.975)
    
    mse = np.sum(residuals**2) / (x.shape[0] - 2)
    
    x_size = x.shape[0]
    x_mean = x.mean()
    
    prediction_margin = t_multiplier * np.sqrt(mse * (1 + 1 / x_size + (x_new - x_mean)**2 / (np.sum((x - x_mean)**2) / (x_size - 1))))
    
    return prediction_margin
```


```python
x_new = np.linspace(-4, 8, 1000)
y_estimated_new = beta_0 + x_new * beta_1

mean_margin = mean_response_interval(x_new, x, residuals)
prediction_margin = prediction_interval(x_new, x, residuals)

# Confidence Interval
plt.plot(x_new, y_estimated_new + mean_margin)
plt.plot(x_new, y_estimated_new - mean_margin,)

# Prediction Interval
plt.plot(x_new, y_estimated_new + prediction_margin)
plt.plot(x_new, y_estimated_new - prediction_margin)

# Data Points
plt.scatter(x, y)

# Fitted Line
plt.plot(x_new, y_estimated_new)

```


    
![image alt text]({static}../images/linear_regression_5.png)
    


```python
x_new_single = 1
y_predicted_x_new = beta_0 + beta_1 * x_new_single
prediction_margin = prediction_interval(x_new_single, x, residuals)
print(f"Prediction for {x_new_single} => {y_predicted_x_new:.2f} ± {prediction_margin:.2f}")

x_new_single = 10
y_predicted_x_new = beta_0 + beta_1 * x_new_single
prediction_margin = prediction_interval(x_new_single, x, residuals)
print(f"Prediction for {x_new_single} => {y_predicted_x_new:.2f} ± {prediction_margin:.2f}")

x_new_single = 30
y_predicted_x_new = beta_0 + beta_1 * x_new_single
prediction_margin = prediction_interval(x_new_single, x, residuals)
print(f"Prediction for {x_new_single} => {y_predicted_x_new:.2f} ± {prediction_margin:.2f}")

print(f"95% CI for Residuals: {residuals.mean():.2f} ± {residuals.mean() + 2 * residuals.std():.2f}")
```

Prediction for 1 => 10.56 ± 56.52

Prediction for 10 => 107.54 ± 402.09

Prediction for 30 => 323.07 ± 1199.95

95% CI for Residuals: -0.00 ± 40.16
    
<br>

From the previous analysis, it can be concluded that any prediction constructed must be reported as a value with a confidence interval, rather than as a single number. Moreover, as the value of $x_{new}$ moves away from the center of the explanatory variables (in this case the mean), the prediction interval grows wider, making the model not suitable for extrapolation, but rather for interpolation (with the corresponding CI). Then, interpolated values are much more reliable than are extrapolated values.

# Multiple Linear Regression

A Multiple Linear Regression (MLR) is a linear approach to modeling the relationship between a dependent variable $y$ and multiple independent variables $X$.

Let $X$ be an $n \times m$ matrix, where $n$ is the number of observations and $m$ is the number of features, and $y$ be an $n \times 1$ vector, the linear regression model assumes a linear relationship between $X$ and $y$. Mathematically, this relationship is modeled as

$$y_i = \beta_0 + \beta_1 x_{i1} + \dotsc + \beta_m x_{im} + \epsilon_i$$

where $\beta_0$ is the intercept, $\beta_1, \cdots, \beta_m$ are the coefficients, and $\epsilon$ is the error, an unobserved random variable that adds noise to the linear relationship between the dependent and the independent variables.

In matrix notation, the model is described by

$$Y = X^T \beta + \epsilon$$

where

$$Y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix} \qquad X= \begin{pmatrix} x_1^T \\ x_2^T \\\vdots \\ x_n^T \end{pmatrix} = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1m} \\ 1 & x_{21} & \cdots & x_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{nm} \end{pmatrix} \qquad \beta = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_m \end{pmatrix} \qquad \epsilon = \begin{pmatrix} \epsilon_1 \\ \epsilon_{2} \\ \vdots \\ \epsilon_n \end{pmatrix}$$

It can be observed that an extra column of $1$s is added to $X$, to vouch for the intercept coefficient.

## Ordinary Least Squares Multiple

The Ordinary Least Squares for multiple variables can be written as

$$\hat{\beta} = \left( X^T X \right)^{-1} X^Ty$$

For a full demonstration, relate to this [website](https://en.wikipedia.org/wiki/Ordinary_least_squares#Matrix/vector_formulation).


```python
def ols_multiple(X, y):
    intercept_matrix = np.ones(X.shape[0])
    X_with_intercept = np.column_stack((intercept_matrix, X))
    betas = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    intercept = betas[0]
    coefficients = betas[1:]
    return intercept, coefficients
```

## MLS Implementation


```python
def multiple_liner_regression_fit(X, y):
    intercept, coefficients = ols_multiple(X, y)
    return intercept, coefficients
```


```python
def multiple_linear_regression_predict(X, intercept, coefficients):
    predictions = intercept + X @ coefficients
    return predictions
```

## MLS Application

**Data set**


```python
boston = datasets.load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Train**


```python
intercept, coefficients = multiple_liner_regression_fit(X_train, y_train)
print(f'β0: {intercept:.3f}')
for i, c in enumerate(coefficients):
    print(f'β{i+1}: {c:.3f}')
```


$\beta_0: 30.247$

$\beta_1: -0.113$

$\beta_2: 0.030$

$\beta_3: 0.040$

$\beta_4: 2.784$

$\beta_5: -17.203$

$\beta_6: 4.439$

$\beta_7: -0.006$

$\beta_8: -1.448$

$\beta_9: 0.262$

$\beta_10: -0.011$

$\beta_11: -0.915$

$\beta_12: 0.012$

$\beta_13: -0.509$
    
<br>

**Test**


```python
y_estimated = multiple_linear_regression_predict(X_test, intercept, coefficients)
rmse = mse(y_test, y_estimated, squared=False)
```

RMSE: 4.9286
    
<br>

# References

[Linear Regression - Wikipedia](https://en.wikipedia.org/wiki/Linear_regression){: target="_blank"}

[Simple Linear Regression - Wikipedia](https://en.wikipedia.org/wiki/Simple_linear_regression){: target="_blank"}

[Deriving the least squares estimators - jbstatistics](https://www.youtube.com/watch?v=ewnc1cXJmGA){: target="_blank"}

[The four assumptions of linear regression - statology](https://www.statology.org/linear-regression-assumptions/){: target="_blank"}

[Testing the assumptions of linear regression - people.duke.edu](http://people.duke.edu/~rnau/testing.htm){: target="_blank"}

[What is wrong with extrapolation? - stats.stackexchange](https://stats.stackexchange.com/questions/219579/what-is-wrong-with-extrapolation){: target="_blank"}

[Confidence and prediction intervals for SLR - stat.duke.edu](http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf){: target="_blank"}

[Prediction Interval for a New Response - online.stat.psu.edu](https://online.stat.psu.edu/stat501/lesson/3/3.3){: target="_blank"}
