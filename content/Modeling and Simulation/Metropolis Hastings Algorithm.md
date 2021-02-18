Title: Metropolis-Hastings Algorithm
Date: 2020.10.02
Summary: Explanation and Implementation of MCMC algorithm Metropolis-Hastings. 

**Notebook written by Gastón Amengual.**

<hr>

```python
import pymc3 as pm
```

The **Metropolis-Hastings algorithm** is a Markov Chain Monte Carlo method for obtaining a sequence of random samples from a probability distribution (henceforth the target distribution), from which direct sampling is difficult. This sequence can be used to approximate the distribution or to compute an integral, even if the normalizing constant is unknown.

A **normalizing constant** is a constant by which an everywhere non-negative function must be multiplied so the area under its graph is 1, to make it a probability density function or a probability mass function. Bayes' theorem states that the posterior probability measure is proportional to the product of the prior probability measure and the likelihood function. *Proportional to* implies that one must multiply or divide by a normalizing constant to assign measure 1 to the whole space to get a probability measure. 

$$P(H_0|D) = \dfrac{P(D|H_0)P(H_0)}{P(D)}$$

Since $P(D)$ is difficult to calculate, an alternative way to describe this relationship is as one of proportionality:

$$P(H_0|D) \propto P(D|H_0)P(H_0)$$

Since $P(H_0|D)$ is a probability, the sum over all possible (mutually exclusive) hypotheses should be 1, leading to the conclusion that

$$P(H_0|D) = \dfrac{P(D|H_0)P(H_0)}{\sum_{i}P(D|H_i)P(H_i)}$$

where $P(D) = \sum_{i}P(D|H_i)P(H_i)$ is the normalizing constant and the reciprocal of the value. For continuous distributions,

$$P(H_0|D) = \dfrac{P(D|H_0)P(H_0)}{\int P(D|H_i)P(H_i)}$$

## Intuition

The Metropolis–Hastings algorithm can draw samples from any probability distribution $P(\theta)$, provided that a function $g(\theta)$ proportional to the density of $P$ is known, and the values of $g(\theta)$ can be calculated. The requirement that $g(\theta)$ must only be proportional to the density rather than exactly equal to it makes the algorithm useful, because calculating the necessary normalization factor is often extremely difficult in practice.

$$P(\theta) \propto g(\theta)$$

The algorithm works by generating a sequence of sample values in such a way that, as more and more sample values are produced, the distribution of values more closely approximates the target distribution $P(\theta)$. These sample values are produced iteratively, with the distribution of the next sample being dependent only on the current sample value (thus making the sequence of samples into a Markov chain). Specifically, at each iteration, the algorithm picks a candidate for the next sample value based on the current sample value. Then, with some probability, the candidate is either accepted (in which case the candidate value is used in the next iteration) or rejected (in which case the candidate value is discarded, and current value is reused in the next iteration). The probability of acceptance is determined by comparing the values of the function $g(\theta)$ of the current and candidate sample values with respect to the desired distribution $P(\theta)$.

## Proposal distribution
The proposal distribution $q(\theta^* \mid \theta_{i-1})$ is the candidate generating distribution from which the candidates are sampled, and to approaches can be considered:

* $q$ does not depends on the previous iteration’s value of $\theta$, for example, if $q(\theta^*)$ is always the same distribution. In this case, $q(\theta)$ should be as similar as possible to $p(\theta)$.

* $q$ depends on the previous iteration (Random-Walk Metropolis-Hastings), and it is centered on $\theta_{i-1}$. For instance, it might be a normal distribution with mean $\theta_{i-1}$. Because the normal distribution is symmetric, $q(\theta^* \mid \theta_{i-1}) = q(\theta_{i-1} \mid \theta^*)$. Thus, when the candidate is drawn from a normal with mean $\theta_{i-1}$ and constant variance, the acceptance ratio is $\alpha = g(\theta^*) / g(\theta_{i-1})$.

## Acceptance rate
Not all candidate draws are accepted by the algorithm, causing the Markov chain to remain at a certain current state for many iterations. How often it is desired to accept candidates depends on the type of algorithm used. 

If $p(\theta)$ is approximated with $q(\theta^*)$ and candidates are always drawn from $q$, accepting candidates often is good, as it means that $q(\theta^*)$ is approximating $p(\theta)$ well. However, it may still be wanted for $q$ to have a larger variance than $p$ and see some rejection of candidates as an assurance that $q$ is covering the space well.

On the other hand, a high acceptance rate for the Random-Walk Metropolis-Hastings sampler is not preferable. If the random walk takes too small of steps, it will accept often, but will take a very long time to fully explore the posterior. If the random walk is taking too large of steps, many of its proposals will have low probability and the acceptance rate will be low, wasting many draws. Ideally, a random walk sampler should accept somewhere between $23\%$ and $50\%$ of the candidates proposed.

## Algorithm

1. Select an initial value $\theta_0$.
2. For $i = 1, \ldots, m$, repeat the following steps:
    * Draw a candidate sample $\theta^*$ from a proposal distribution $q(\theta^* \mid \theta_{i-1})$.
    * Compute the ratio $$\alpha = \frac{g(\theta^*) / q(\theta^* \mid \theta_{i-1}) }{g(\theta_{i-1}) / q(\theta_{i-1} \mid \theta^*)} = \frac{g(\theta^*)q(\theta_{i-1} \mid \theta^*)}{g(\theta_{i-1})q(\theta^* \mid \theta_{i-1})} \,$$ .

    * If $\alpha \ge 1$, then set $\theta_i = \theta^*$. If $\alpha < 1$, then set $\theta_i = \theta^*$ with probability $\alpha$, or $\theta_i = \theta_{i-1}$ with probability $1-\alpha$.
    
Steps 2b and 2c act as a correction since the proposal distribution is not the target distribution. At each step in the chain, we draw a candidate and decide whether to “move” the chain there or remain where we are. If the proposed move to the candidate is “advantageous,” $(\alpha \ge 1)$ we “move” there and if it is not “advantageous,” we still might move there, but only with probability $\alpha$. Since our decision to “move” to the candidate only depends on where the chain currently is, this is a Markov chain.

## Example

In a particular industry, it is desired to know the growth $\mu$ of the companies. 

The data $y$ represent the percent change in total personnel from last year to this year for $n=10$ companies, $y = (1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)$.

Although the conjugate prior for $\mu$ would be a normal distribution, a t-distribution is assumed to better reflect the prior beliefs. As it is centered on $0$, there is a $50\%$ chance of the growth being positive or negative.

$$\mu \sim t(0,1,1) \quad \rightarrow \quad \text{Prior Distribution}$$ 

The likelihood is a normal distribution with known variance.

$$y_i | \mu \sim N(\mu,1), \; i=1,...,n \quad \rightarrow \quad \text{Likelihood Distribution}$$

Because this model is not conjugate, the posterior distribution is not in a standard form from which can be easily sampled. To obtain posterior samples, a Markov chain whose stationary distribution is this posterior distribution is set up.

$$p(\mu | y_1, ..., y_n) \propto \prod_{i=1}^{n}[N(\mu,1)] \cdot t(0,1,1) $$

$$p(\mu | y_1, ..., y_n) \propto \prod_{i=1}^{n} \left [ \dfrac{1}{\sqrt{2\pi}}e^{-0.5(y_i-\mu)^2)} \right ] \cdot \dfrac{1}{\pi(1+\mu^2)} $$

$$p(\mu | y_1, ..., y_n) \propto \dfrac{e^{n(\bar{y}\mu-\mu^2/2)}}{1+\mu^2} \quad \rightarrow \quad \text{Posterior distribution}$$

Because posterior distributions include likelihoods (the product of many numbers that are potentially small), $g(\mu)$ might evaluate to such a small number that are considered zero to the computer, causing a problem when evaluating the acceptance ratio. To avoid this problem, the log scale will be used:

$$log(p(\mu | y_1, ..., y_n)) \propto n(\bar{y}\mu-\mu^2/2) - log(1+\mu^2)$$

The candidates will be drawn from a normal proposal distribution $q(\mu) \sim N(\mu_i, 1)$. As $q$ is a symmetric distribution, $\alpha = \dfrac{g(\mu)}{g(\mu_{i-1})}$, $log(\alpha) = log \; g(\mu) - log \; g(\mu_{i-1})$


```python
def prior_distribution(n):
    x = np.linspace(stats.t.ppf(0.01, df=9), stats.t.ppf(0.99, df=9), n)
    y = stats.t.pdf(x, df=9)
    return x, y
```


```python
def proposal_distribution(mean, std):
    return np.random.normal(mean, std)
```


```python
def log_g(mu, n, y_mean):
    return  n * (y_mean * mu - mu**2/2) - np.log(1 + mu**2)
```


```python
data = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9])
```

## Implementation


```python
def metropolis_hastings(initial_mu, candidate_std, num_iterations, n, data_mean):
    
    current_mu = initial_mu
    current_value = log_g(current_mu, n, data_mean)
    
    num_acceptances = 0
    
    mu_chain = [] 

    for i in range(num_iterations):
        # Draw candidate
        candidate_mu = proposal_distribution(mean=current_mu, std=candidate_std)
        candidate_value = log_g(candidate_mu, n, data_mean)
    
        # Calculate alpha
        log_alpha = candidate_value  - current_value
        alpha = np.exp(log_alpha)
        
        # Accept or reject
        if np.random.rand() < alpha:
            current_mu = candidate_mu
            num_acceptances += 1
            current_value = candidate_value
        
        # Add to chain
        mu_chain.append(current_mu)
        
    acceptance_rate = num_acceptances/num_iterations
        
    return np.array(mu_chain), acceptance_rate
```

## Posterior sampling with different STDs


```python
candidate_stds = [0.0005, 0.05, 0.5, 1, 10, 20]
initial_mu = 0
num_iterations = 100000
data_size = len(data)
data_mean = data.mean()

posterior_samples = []

for candidate_std in candidate_stds:  
    mu_chain, acceptance_rate = metropolis_hastings(initial_mu, candidate_std, num_iterations, data_size, data_mean)
    
    posterior_samples.append(mu_chain)    
    
    pm.plot_trace(mu_chain, figsize=(16,5))
```


    
![image alt text]({static}../images/metropolis_hastings_1.png)
    



    
![image alt text]({static}../images/metropolis_hastings_2.png)
    



    
![image alt text]({static}../images/metropolis_hastings_3.png)
    



    
![image alt text]({static}../images/metropolis_hastings_4.png)
    



    
![image alt text]({static}../images/metropolis_hastings_5.png)
    



    
![image alt text]({static}../images/metropolis_hastings_6.png)
    


The distribution generated with std $= 1$ and acceptance rate $= 0.57$ is chosen as the posterior distribution for $\mu$.


```python
# Data
sns.kdeplot(data, color='rebeccapurple', label='Data Distribution')

# Prior
x_prior, y_prior = prior_distribution(1000)
plt.plot(x_prior, y_prior, color='olivedrab', label='Prior Distribution')

# Posterior
burn_in = int(num_iterations*0.1)
posterior = posterior_samples[3][burn_in:]
sns.kdeplot(posterior, color='darkcyan', label='Posterior Distribution')
```


    
![image alt text]({static}../images/metropolis_hastings_7.png)
    
<hr>

# References

Bayesian Statistics: Techniques and Models - University of California Santa Cruz - Coursera 

https://www.wikiwand.com/en/Metropolis%E2%80%93Hastings_algorithm

https://www.wikiwand.com/en/Normalizing_constant
