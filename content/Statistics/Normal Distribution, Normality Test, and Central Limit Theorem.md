Title: Central Limit Theorem and Normality Test 
Date: 2021.01.21
Summary: Application of Normality Test to normal-like distributions. Explanation of the Central Limit Theorem, and verification with Normality Test for different combinations of sums of uniform, exponential, and normal samples of different sizes.




```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from tqdm.notebook import tqdm
```


```python
plt.style.use("bmh")
config = {'figure.figsize': (16,5), 
          'axes.titlesize': 18, 
          'axes.labelsize': 14, 
          'lines.linewidth': 2, 
          'lines.markersize': 10, 
          'xtick.labelsize': 10,
          'ytick.labelsize': 10, 
          'axes.prop_cycle': plt.cycler(color=["mediumpurple", "saddlebrown", "darkcyan", "olivedrab", "darkseagreen", "darkkhaki", "darkgoldenrod", "deepskyblue", "firebrick", "palevioletred"]),}
plt.rcParams.update(config)
```

*This article uses concepts related to Hypothesis Testing, such as p-values and null hypothesis. However, this knowledge should not determine the understanding the article, as it will be only a tool to prove or disprove a given statement of a certain topic. That statement or null hypothesis will be proposed, and the p-values will accept or reject the hypothesis. For more information on Hypothesis Testing, please visit https://en.wikipedia.org/wiki/Statistical_hypothesis_testing.*

# 1 Normality Test

Normality tests are used to determine if a data set is well-modeled by a normal distribution and to compute how likely it is for a random variable underlying the data set to be normally distributed. The scipy.stats.normaltest function will be used, which tests the null hypothesis that a sample comes from a normal distribution, and it is based on D’Agostino and Pearson’s test that combines skew and kurtosis to produce an omnibus test of normality.

$1000$ p-values for samples with same size will be taken applying the normal test. The acceptance percentage will be calculated as the proportion of those p-values that are less than $\alpha = 0.5$. That proportion is the number of times the given null hypothesis was rejected.


```python
def normal_test(distribution, name):
    print(f'Null Hypothesis: The {name} Distribution is a Normal Distribution')

    sizes = [30, 50, 100, 200, 500, 1000, 2000, 5000]
    for size in sizes:
        p_values = []
        for _ in range(1000):
            data = distribution(size)
            _, p_value = stats.normaltest(data)
            p_values.append(p_value)

        acceptance = np.mean(np.array(p_values) < 0.05)

        print(f'From 1000 samples of {size:4.0f} elements, {acceptance * 100:2.2f}% of the time the null hypothesis was rejected')
```

## 1.1 Normal-looking Custom Probability Function

In many occasions, one can mistakenly suggest that a data sample follows a normal distribution just because its visualization is *normal-like*. However, this assumption should not be taken lightly, and must be statistically verified. The following distribution was built to resemble a normal distribution and to check the efficiency of the normal test.  


```python
def custom_distribution(size):
    return np.arctanh(np.random.rand(size)) * np.random.choice([-1, 1], replace=True, size=size)
```

To verify that the function is a probability distribution, the area under its curve must be equal to 1.


```python
size = 100000
histogram = np.histogram(custom_distribution(size), bins=50, density=True)
sum_area_rectangles = (np.diff(histogram[1]) * histogram[0]).sum()
print(f'The area under the curve equals {sum_area_rectangles:.3g}.')

assert np.isclose(sum_area_rectangles, 1)

print('It is a probability distribution.')
```

    The area under the curve equals 1.
    It is a probability distribution.
    


```python
plt.hist(custom_distribution(size), bins=50, density=True, label='Custom Distribution')
plt.hist(np.random.normal(size=size), color="firebrick", density=True, alpha=0.5, bins=30, label='Normal Distribution')
plt.title('Comparison of Custom and Normal distributions')
plt.xlim(-4,4)
plt.legend()
plt.tight_layout()
plt.show()
```


    
![image alt text]({static}../images/normal_distribution_normality_test_and_central_limit_theorem_2.png)
    



```python
normal_test(custom_distribution, 'Custom Distribution')
```

    Null Hypothesis: The Custom Distribution Distribution is a Normal Distribution
    From 1000 samples of   30 elements, 18.00% of the time the null hypothesis was rejected
    From 1000 samples of   50 elements, 25.20% of the time the null hypothesis was rejected
    From 1000 samples of  100 elements, 36.40% of the time the null hypothesis was rejected
    From 1000 samples of  200 elements, 56.30% of the time the null hypothesis was rejected
    From 1000 samples of  500 elements, 85.30% of the time the null hypothesis was rejected
    From 1000 samples of 1000 elements, 98.80% of the time the null hypothesis was rejected
    From 1000 samples of 2000 elements, 100.00% of the time the null hypothesis was rejected
    From 1000 samples of 5000 elements, 100.00% of the time the null hypothesis was rejected
    

As can be observed, beginning in fairly small data samples ($500$) the normal distribution fails to model the custom distribution samples most of the times, and fails completely for sample size bigger than approximately $2000$.

## 1.2 Customized Beta Distribution

To verify the effectiveness of the normal test, a normal-like beta distribution will be tested. After experimentation, it was found that $(Beta(5, 5) \cdot 6) - 3)$ resembles a normal distribution.


```python
def custom_beta(size):
    return np.random.beta(5, 5, size=size) * 6 - 3
```


```python
size = 100000
plt.hist(custom_beta(size), bins=50, density=True, color='Salmon', label='Custom Beta Distribution')
plt.hist(np.random.normal(size=size), color="Olivedrab", density=True, alpha=0.5, bins=50, label='Normal Distribution')
plt.title('Comparison of Custom Beta and Normal distributions')
plt.xlim(-4,4)
plt.legend()
plt.tight_layout()
plt.show()
```


![image alt text]({static}../images/normal_distribution_normality_test_and_central_limit_theorem_3.png)
    



```python
normal_test(custom_beta, 'Custom Beta')
```

    Null Hypothesis: The Custom Beta Distribution is a Normal Distribution
    From 1000 samples of   30 elements, 3.70% of the time the null hypothesis was rejected
    From 1000 samples of   50 elements, 4.40% of the time the null hypothesis was rejected
    From 1000 samples of  100 elements, 9.70% of the time the null hypothesis was rejected
    From 1000 samples of  200 elements, 21.00% of the time the null hypothesis was rejected
    From 1000 samples of  500 elements, 65.90% of the time the null hypothesis was rejected
    From 1000 samples of 1000 elements, 97.00% of the time the null hypothesis was rejected
    From 1000 samples of 2000 elements, 100.00% of the time the null hypothesis was rejected
    From 1000 samples of 5000 elements, 100.00% of the time the null hypothesis was rejected
    

It can be concluded that from a sample size of $1000$, the normal distribution fails to model the custom beta distribution most of the times.

# 2 Central Limit Theorem

Let $X_1, \dotsc, X_n$ be a random sample of size $n$, i.e. a sequence of independent and identically distributed random variables from a distribution of expected value $\mu$ and finite variance $\sigma^2$. The sample average of the random variables is 

$$\bar{X}_n = \dfrac{X_1 + \dotsc + X_n}{n}$$

By the law of large numbers, the sample averages $\bar{X}_n $converge almost surely to the expected value $\mu$ as $n \to \infty$. The Central Limit Theorem describes the size and the distributional form of the stochastic fluctuations around the deterministic number $\mu$ during this convergence. It states that, as $n$ gets larger, the distribution of the difference between the sample average $\bar{X}_n$ and its limit $\mu$, when multiplied by the factor $\sqrt{n}$, that is $\sqrt{n} ( \bar{X}_n -\mu)$, approximates the normal distribution with mean $0$ and variance $\sigma^2$. For large enough $n$, the distribution of $\bar{X}_n$ is close to the normal distribution with mean $\mu$ and variance $\sigma^{2}/n$. The usefulness of the theorem is that the distribution of $\sqrt{n} ( \bar{X}_n -\mu)$ approaches normality regardless of the shape of the distribution of the individual $X_i$. This fact holds especially true for sample sizes over $30$. As more samples are taken, especially large ones, the graph of the sample means looks more like a normal distribution. 

The Central Limit Theorem partially explains the prevalence of normal distributions in the natural world. Most characteristics of animals and other life forms are affected by a large number of genetic and environmental factors whose effects are additive. As the characteristics measure are the sum of a large number of small effects, so their distribution tends to be normal.

## 2.1 Die Roll Example

Suppose a die is rolled. The expected value of the dice is $E[X] = \sum_{i} x_i \cdot P(X = x_i) = \frac{1}{6} \cdot (1 + 2 + 3 + 4 + 5 + 6) = 3.5$. Although it is not possible to obtain a $3.5$ on a roll of a die, with an increase in the number of die rolls, the average of the die rolls would be close to $3.5$.


```python
fig, axes = plt.subplots(2, 3, figsize=(16, 6))

bins = [15, 15, 18, 25, 35, 35]
sizes = [5, 30, 100, 500, 1000, 10000]
colors = ["mediumpurple", "saddlebrown", "deepskyblue", "olivedrab", "darkseagreen", "darkkhaki", "darkgoldenrod", "darkcyan"]

for ax, size, bins, color in zip(axes.reshape(-1), sizes, bins, colors):
    means = np.random.randint(1, 7, size=(size, 10)).mean(axis=1) # Calculate mean of size samples of 10 uniform distibuted number
    ax.hist(means, bins=15, color=color, density=True)
    ax.set_title(f'{size} samples - $E[X]={means.mean():.3g}$', fontsize=15)
    ax.set_axisbelow(True)

plt.suptitle('Histogram of Expected Value of Die Rolls', fontsize=20)
plt.tight_layout()
plt.show()
```


    
![image alt text]({static}../images/normal_distribution_normality_test_and_central_limit_theorem_4.png)
    


It can be observed that with a small number of samples, the histogram does not present a definite pattern. However, by increasing the sample size, the sampling distribution starts to resemble a normal distribution (demonstrated by Central Limit Theorem), and the most frequent average of $10$ die rolls is approximately $3.5$.

## 2.2 Normal Tests for CLT

There are two parameters that can be varied when applying the Central Limit Theorem. $n$, the number of i.i.d. variables used to calculate the sample mean (10 in the previous example), and $r$, the number of sample means that are taken to form the sample mean distribution (5, 30, ..., 10000 in the previous example).

In this section, the normal test will be applied for different combinations of $r$ and $n$ for the Uniform, Normal, and Exponential distributions, following the following **Null Hypothesis**: 

*The number of $r$ sample means of $n$ i.i.d. variables of some distribution follows a Normal Distribution.*

For every $r-n$ combination, 1000 p-values will be calculated, along with the mean of how many of them were lower than the significance level.


```python
def normal_test_varying_r_n(distribution, significance=0.05, rs=None, ns=None):
    
    if rs is None:
        rs = [30, 50, 100, 500, 1000, 5000, 10000]
    if ns is None:
        ns = [8, 30, 100, 500, 1000, 5000]

    df = pd.DataFrame(index=rs, columns=ns)

    for r in tqdm(rs):
        for n in ns:
            p_values = []

            # Get p-value for 1000 samples of size (r, n) and calculate average
            for _ in range(1000):
                data = distribution(size=(r, n)).mean(axis=1)
                _, p_value = stats.normaltest(data)
                p_values.append(p_value)

            df[n][r] = np.mean(np.array(p_values) < significance) * 100
            
    return df
```


```python
def color_less_alpha(val):
    alpha = 0.05
    if val <= alpha * 100:
        bg_color = 'forestgreen'
    elif val < alpha * 3 * 100:
        bg_color = 'darkorange'
    else:
        bg_color = 'firebrick'
    return f'background-color: {bg_color}; color: white; font-weight: bold;'
```


```python
distributions = [np.random.uniform, np.random.exponential, np.random.normal]
for distribution in distributions:
    df = normal_test_varying_r_n(distribution)
    df.to_csv(f'{distribution.__name__}.csv')
```

In the following matrices, $r$ will correspond to the rows, and $n$ to the columns, and each cells shows the percentage of times the null hypothesis was rejected. The pattern is as follows:

<ul>
    <li style="font-weight: bold; color:forestgreen;">Null Hypothesis was rejected less than 5% of the times</li>
    <li style="font-weight: bold; color:darkorange;">Null Hypothesis was rejected between 5% and 15% of the times</li>
    <li style="font-weight: bold; color:firebrick;">Null Hypothesis was rejected more than 15% of the times</li>
</ul>


```python
df = pd.read_csv('normal_uniform_samples.csv', index_col=0)
df.rename_axis(index='$r$', columns='$n$', inplace=True)
display(df.style.applymap(color_less_alpha).format("{:.5g}").set_caption("Uniform Samples").set_table_styles([{'selector': 'caption', 'props': [('font-size', '1.5rem'), ('font-weight', 'bold')]}]))

df = pd.read_csv('normal_exponential_samples.csv', index_col=0)
df.rename_axis(index='$r$', columns='$n$', inplace=True)
display(df.style.applymap(color_less_alpha).format("{:.5g}").set_caption("Exponential Samples").set_table_styles([{'selector': 'caption', 'props': [('font-size', '1.5rem'), ('font-weight', 'bold')]}]))
        
df = pd.read_csv('normal_normal_samples.csv', index_col=0)
df.rename_axis(index='$r$', columns='$n$', inplace=True)
display(df.style.applymap(color_less_alpha).format("{:.5g}").set_caption("Normal Samples").set_table_styles([{'selector': 'caption', 'props': [('font-size', '1.5rem'), ('font-weight', 'bold')]}]))
```

![image alt text]({static}../images/normal_distribution_normality_test_and_central_limit_theorem_5.png)


It can be concluded that the samples generated for most of the combinations of the uniform and normal distribution were normally distributed most of the times, while for the exponential distribution, larger values of $r$ did not change the acceptance proportion (they were mostly rejected), being the number of $n$ the most critical parameter for the sample to be follow a normal distribution.

In summary, not many variables follow a normal distribution, but, thanks to Central Limit Theorem, this distribution can be found in many natural phenomena when a sufficiently large sample is given, simplifying their study and analysis.

# References

<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html">https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html</a>

<a href="https://en.wikipedia.org/wiki/Normality_test">https://en.wikipedia.org/wiki/Normality_test</a>

<a href="https://en.wikipedia.org/wiki/Central_limit_theorem">https://en.wikipedia.org/wiki/Central_limit_theorem</a>