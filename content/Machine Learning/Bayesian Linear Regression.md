Title: Bayesian Linear Regression
Date: 2021.01.30
Summary: Definition. Frequentists vs Bayesians. Comparison to Linear Regression.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

import pymc3 as pm
```


# Frequentists vs Bayesians

### Frequentist approach 

1. The true parameters are fixed. The data is variable.

2. It fits the variable data to the model. That requires the meeting of the assumptions, as, if they are not met, the model does not guarantees correct results. 

3. The output are single fixed parameters.

4. For a Simple Linear Regression, the mean response is a single line.

5. For each parameter, a confidence interval can be calculated. This confidence interval is not part of the output of the model, and it focuses on the variability of the data: when taking many samples, 95% of the time this true fixed parameter will be included in the interval. 

### Bayesian approach

1. There are no true parameters. The parameters are not fixed, but variable, and the data is fixed.

2. It fits the parameters and the model to the fixed data. This introduces a more experimental approach in which assumptions can be changed or relaxed (such as normality or heteroscedasticity), as it is the model that must adapt to the data, not the data to the model.

3. The output is a probability distribution for each parameter. It does not try to find the single best value of the parameters.

4. For a Simple Linear Regression, the mean response is a set of lines

5. Each parameter distribution has its corresponding credible interval (94% is generally used). This interval is a part of the output (such as the mean or the Std.), as it is the most of the area under the curve of the distribution, not and extra calculation, and focuses on the variability of parameters. 

There are two critical advantages of Bayesian estimation. With priors any prior knowledge can be quantified by placing priors on the parameters. For instance, if a parameter is thought likely to be small, a prior with more probability mass on low values can be chosen. Also, uncertainty can be quantified, as one gets not a single estimate of the parameter, but instead a complete posterior distribution about how likely different values of these parameters are. For few data points, the uncertainty in the parameters will be very high and very wide posteriors will be obtained.

# Toy Dataset


```python
x, y = make_regression(n_samples=5000, n_features=1, noise=20, random_state=0)
x = x.flatten().reshape(-1, 1)
y = y.reshape(-1, 1)
plt.scatter(x, y)
```

    
![image alt text]({static}../images/bayesian_linear_regression_1.png)
    


# Frequentist Linear Regression

A standard linear regression is defined as

$$Y = \beta_1 X + \beta_0 + \epsilon, \; \text{with} \; \epsilon_i \sim N(0, \sigma^2)$$

The coefficients can be estimated using Ordinary Least Squares (OLS) or Maximum Likelihood.


```python
reg = LinearRegression().fit(x, y)

beta_1 = reg.coef_[0][0]
beta_0 = reg.intercept_[0]
y_estimated = beta_1 * x + beta_0

plt.scatter(x, y, s=30)
plt.plot(x, y_estimated)
```


    
![image alt text]({static}../images/bayesian_linear_regression_2.png)
    


In the above frequentist estimation process, the output of the linear regression are single fixed values for both the model parameters ($\beta_0 = -0.22$ and $\beta_1 = 10.78$) and the predictions (the predicted value for $x_i = 5.8$ is $\hat{y}_i = 62.28$).


```python
def mean_response_95_interval(x_new, x, residuals):
    t_multiplier = stats.t(x.shape[0] - 2).ppf(0.975)
    mse = np.sum(residuals**2) / (x.shape[0] - 2)
    x_size = x.shape[0]
    x_mean = x.mean()
    mean_margin = t_multiplier * np.sqrt(mse * (1 / x_size + (x_new - x_mean)**2 / (np.sum((x - x_mean)**2) / (x_size - 1))))
    return mean_margin
```


```python
residuals = y - y_estimated
x_new = 2
y_estimated_new = beta_0 + x_new * beta_1

mean_margin = mean_response_95_interval(x_new, x, residuals)
print(f'Confidence interval for x=2: ({y_estimated_new - mean_margin:.2f}, {y_estimated_new + mean_margin:.2f})')
```

Confidence interval for x=2: (-59.17, 101.84)
    

# Bayesian Linear Regression

Bayesian linear regression is an approach to linear regression in which the statistical analysis is undertaken within the context of Bayesian inference.

The expression $Y = \beta_1 X + \beta_0 + \epsilon, \; \text{with} \; \epsilon_i \sim N(0, \sigma^2)$ can be written as

$$Y \sim N(\beta_1 X + \beta_0, \sigma^2)$$

where

$$\sigma^2 \sim Exp(1)$$

Put in words, $Y$ is a normally distributed random variable with mean $\beta_1 X + \beta_0$ (that is, the quantity predicted) and some standard deviation $\sigma$.


```python
with pm.Model() as linear_regression_model:
    
    # Priors
    beta_1 = pm.Normal('beta_1', 0, 1)
    beta_0 = pm.Normal('beta_0', 0, 1)
    s = pm.Exponential('sigma', 1)
    
    y_ = beta_1 * x + beta_0
    
    likelihood = pm.Normal('likelihood', y_, s, observed=y)
    
    # Inference: draw posterior sample using NUTS sampling
    trace = pm.sample(draws=1000, tune=8000)
```



```python
beta_1_samples = trace['beta_1']
beta_0_samples = trace['beta_0']
s_samples = trace['sigma']
```


```python
fig, axes = plt.subplots(3, 2, figsize=(16, 8))

pm.plot_posterior(beta_0_samples, ax=axes[0][0])
axes[0][1].plot(beta_0_samples)

pm.plot_posterior(beta_1_samples, ax=axes[1][0])
axes[1][1].plot(beta_1_samples)

pm.plot_posterior(s_samples, ax=axes[2][0])
axes[2][1].plot(s_samples)

plt.suptitle('Posterior Distribution for Parameters', fontsize=20)
plt.tight_layout()
plt.show()
```


    
![image alt text]({static}../images/bayesian_linear_regression_3.png)
    


The mean of the parameters distributions are very similar to the fixed parameters estimated using OLS.

## Mean response for new point


```python
for beta_0, beta_1 in zip(beta_0_samples, beta_1_samples):
    plt.plot(x, beta_0 + beta_1*x, color='gray', alpha=0.8)
    
plt.scatter(x, y, s=20)
```


    
![image alt text]({static}../images/bayesian_linear_regression_4.png)
    


Once more, the mean response of a new x point is not a single point, but a credible interval.


```python
x_new = 2
mean_response_for_x_new = beta_0_samples + beta_1_samples * x_new
_003_quantile = np.quantile(mean_response_for_x_new, 0.03)
_097_quantile = np.quantile(mean_response_for_x_new, 0.97)

plt.hist(mean_response_for_x_new)
plt.hlines(0.05, _003_quantile, _097_quantile)
plt.scatter(_003_quantile)
plt.scatter(_097_quantile)
```


    
![image alt text]({static}../images/bayesian_linear_regression_5.png)
    


# References

[Bayesian Linear Regression - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_linear_regression){: target="_blank"}

[GLM: Linear regression](https://docs.pymc.io/notebooks/GLM-linear.html){: target="_blank"}

[Bayesian Linear Regression in Python via PyMC3](https://towardsdatascience.com/bayesian-linear-regression-in-python-via-pymc3-ab8c2c498211){: target="_blank"}

[Introduction to Bayesian Linear Regression](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7){: target="_blank"}