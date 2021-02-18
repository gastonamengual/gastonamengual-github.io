Title: Confidence Intervals and Bootstrapping
Date: 2021.01.21
Summary: Explanation of Confidence Intervals, Bootstrapping, and comparison.


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
plt.style.use("bmh")
config = {'figure.figsize': (16, 5),
          'axes.titlesize': 18, 
          'axes.labelsize': 14, 
          'lines.linewidth': 2, 
          'lines.markersize': 10, 
          'xtick.labelsize': 10,
          'ytick.labelsize': 10, 
          'axes.prop_cycle': plt.cycler(color=["mediumpurple", "saddlebrown", "darkcyan", "olivedrab", "darkseagreen", "darkkhaki", "darkgoldenrod", "deepskyblue", "firebrick", "palevioletred"]),}
plt.rcParams.update(config)
```

# Confidence Intervals

A confidence interval is a type of estimate computed from the statistics of the observed data, and were introduced by Jerzy Neyman in 1937. This proposes a range of plausible values for an unknown parameter, like the population mean. The interval has an associated confidence level that the true parameter is in the proposed range. 

The confidence level represents the frequency of possible confidence intervals that contain the true value of the unknown population parameter. If confidence intervals are constructed using a given confidence level from an infinite number of independent sample statistics, the proportion of those intervals that contain the true value of the parameter will be equal to the confidence level. 

If an experiment in which a confidence interval is calculated is repeated infinite times, a certain percentage of the experiments will capture the population parameter in their confidence intervals. Most of the times, frequentists choose an interval that captures the population parameter 95% of the time, which be used in the following examples.

## Interpretation

* The confidence interval can be expressed in terms of repeated samples: Were this procedure to be repeated on numerous samples, the fraction of calculated confidence intervals (which would differ for each sample) that encompass the true population parameter would tend toward 95%.

* The confidence interval can be expressed in terms of a single sample: There is a 90% probability that the calculated confidence interval from some future experiment encompasses the true value of the population parameter." It is crucial to note that this probability statement is about the confidence interval, not about the population parameter. This considers the probability associated with a confidence interval from a pre-experiment point of view, and it is known, before the actual experiment is carried out, that the single interval calculated has a particular chance of covering the true but unknown value.

* The explanation of a confidence interval can amount to something like: "The confidence interval represents values for the population parameter for which the difference between the parameter and the observed estimate is not statistically significant at the 10% level".

## Calculation

It is assumed that the samples are drawn from an (approximately) normal distribution.

1. Identify the sample mean $\bar{x}$.

**Population standard deviation $\sigma$ is known**
 
2. $z^* = - \Phi^1 \left( \dfrac{\alpha}{2} \right)$, where $\Phi$ is the CDF of the standard normal distribution (used as critical value), and the confidence level $C = 100(1 - \alpha)%$.

$C \rightarrow Z^*$

$99\% \rightarrow 2.576$

$98\% \rightarrow 2.326$

$95\% \rightarrow 1.96$

$90\% \rightarrow 1.645$

3. The confidence interval is calculated as

$$\left( \bar{x} - z^*\dfrac{\sigma}{\sqrt{n}} \; , \; \bar{x} + z^*\dfrac{\sigma}{\sqrt{n}} \right)$$

**Population standard deviation $\sigma$ is unknown and estimated by the sample standard deviation $s$.**

2. If the population standard deviation is unknown t-values and the sample size is below 30, then the Student's t distribution is used as the critical value. This value is dependent on the confidence level (C) for the test and degrees of freedom. The degrees of freedom are found by subtracting one from the number of observations, n − 1. The critical value is found from the t-distribution table. 

3. The confidence interval is calculated as

$$\left( \bar{x} - t^*\dfrac{s}{\sqrt{n}} \; , \; \bar{x} + t^*\dfrac{s}{\sqrt{n}} \right)$$

## Cup Filling Example

A machine fills cups with a liquid, and is supposed to be adjusted so that the content of the cups is $250 g$ of liquid. As the machine cannot fill every cup with exactly 250 g, the content added to individual cups shows some variation, and is considered a random variable $X$. This variation is assumed to be normally distributed around the desired average of $250 g$, with a standard deviation $\sigma = 2.5 g$. To determine if the machine is adequately calibrated, a sample of $n = 25$ cups of liquid is chosen at random and the cups are weighed. The resulting measured masses of liquid are $X_1, \dotsc, X_{25}$, a random sample from $X$.


```python
mean = 250
std = 2.5
sample_size = 20
z_score = 1.96 # for 95% confidence
num_samples = 1000

contains_population_mean = []
confidence_intervals = []

for i in range(num_samples):
    samples = np.random.normal(mean, std, sample_size)
    
    sample_mean = samples.mean()
    lower_bound = sample_mean - 1.96 * (std / np.sqrt(sample_size))
    upper_bound = sample_mean + 1.96 * (std / np.sqrt(sample_size))
    
    if (mean > lower_bound) and (mean < upper_bound):
        contains_population_mean.append(True)
    else: 
        contains_population_mean.append(False)
        
    confidence_intervals.append([lower_bound, upper_bound])

confidence_level = np.mean(contains_population_mean)*100
confidence_intervals = np.array(confidence_intervals) 
```


```python
not_containing_intervals_index = []
for i, (lower, upper) in enumerate(confidence_intervals):
    if (mean < lower) or (mean > upper):
        not_containing_intervals_index.append(i)

not_containing_intervals = confidence_intervals[not_containing_intervals_index]
```


```python
plt.vlines(np.arange(0, num_samples), confidence_intervals[:,0], confidence_intervals[:,1])

plt.vlines(not_containing_intervals_index, not_containing_intervals[:,0], not_containing_intervals[:,1], color='firebrick')

plt.axhline(mean, color='black', linestyle='--', linewidth=3, label='Population mean')
plt.text(x=num_samples/2, y=245, s=f'95% confidence interval: ({lower_bound:.2f}, {upper_bound:.2f}). With {confidence_level:.3g}% probability, this confidence interval contains the value of the population mean', horizontalalignment='center', fontsize=15)

plt.xlim(0, num_samples)
plt.title(f'{confidence_level:.3g}% of the intervals contain the population mean', fontsize=15)
plt.suptitle(f'{num_samples} days confidence intervals', fontsize=22)

plt.legend()
plt.tight_layout()
plt.show()
```


    
![image alt text]({static}../images/confidence_intervals_and_bootstrapping_1.png)


# Bootstrapping

On the example analyzed previously, both population mean and standard deviation were known. However, in real life that hardly happens, and it is only a single sample that can be worked with, and it is the best and only information possessed about the population.

**Resampling** is a non-parametric method of statistical inference that consists of drawing repeated samples from the original data sample. 

**Bootstrapping** is a resampling method for estimating quantities about a population by averaging estimates from multiple data samples, and can be used to estimate the confidence intervals. Bootstrapping is done with sampling with replacement, that is, the samples are constructed by drawing observations from data sample one at a time and returning them to the sample after they have been chosen, allowing an observation to be included in a sample more than once. 

Although other statistical techniques used to determine confidence intervals assume a known population mean or standard deviation, bootstrapping does not require anything other than a statistically significant sample, i.e. large enough (otherwise may derive in lack of validity to estimate the parameter).

The **steps of bootstrapping** are:

1. Repeat $n$ times:

    1. Draw a sample with replacement of size equal to the original data set size.

    2. Calculate and save the statistic on the sample.

2. Calculate the statistic of the calculated sample statistics.

To sum up, bootstrapping does resampling with replacement maintaining data structure but reshuffling values, extrapolating to the population. It is useful for estimating statistical parameters where data are non-normal, have unknown statistical properties, or lack a standard calculation. Nonetheless, some parameters like variance and standard deviation ah inherently biased (underestimated) in bootstrapping, as extreme values on the edges tend to be rare to appear in bootstrap samples.

Let us continue the previous Cup Filling Example, now calculating the confidence interval with a Bootstrap Resample approach.


```python
original_sample = np.random.normal(mean, std, sample_size)

sample_mean_distribution = []
for i in range(100000):
    bootstrap_sample = np.random.choice(a=original_sample, size=len(original_sample), replace=True)
    sample_mean_distribution.append(bootstrap_sample.mean())
    
sample_mean_distribution = np.array(sample_mean_distribution)
```

The $95\%$ confidence interval is constructed of the $2.25$ and $97.5$ percentiles.  


```python
bootstrap_sorted = np.sort(sample_mean_distribution)
lower_bound = np.percentile(bootstrap_sorted, 2.25)
upper_bound = np.percentile(bootstrap_sorted, 97.5)
mean = sample_mean_distribution.mean()

plt.figure(figsize=(16,4))
plt.hist(sample_mean_distribution, bins=25, density=True)
plt.text(x=mean, y=-0.2, s=f'95% confidence interval: ({lower_bound:.2f}, {upper_bound:.2f})', ha='center', fontsize=17)
plt.title(f'Bootstrapped Resampling Mean Distribution - Mean: {mean:.2f}', fontsize=20)
plt.tight_layout()
plt.show()
```


    
![image alt text]({static}../images/confidence_intervals_and_bootstrapping_2.png)
    


As can be observed, this method shows similar results as those produced by the analytic calculation, and approximates very efficiently the population mean.

# References

https://www.wikiwand.com/en/Confidence_interval

https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/

https://www.thoughtco.com/example-of-bootstrapping-3126155

https://www.statisticshowto.com/resampling-techniques/

Resampling methods (bootstrapping) - Matthew E. Clapham - https://www.youtube.com/watch?v=gcPIyeqymOU 
