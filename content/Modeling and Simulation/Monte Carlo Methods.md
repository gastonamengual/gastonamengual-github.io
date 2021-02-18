Title: Monte Carlo Methods
Date: 2020.09.24
Summary: Includes three applications of the Monte Carlo Methods: Integration, Pi Approximation, and Custom Distribution Sampling.

**Notebook written by Gastón Amengual.**

<hr>

The Monte Carlo Methods are computational algorithms, numerical methods of solving mathematical problems by random sampling (or by the simulation of random variables). They rely on random number generation to solve deterministic problems.

The Law of Large Numbers states that as the number of identically distributed, randomly generated variables increases, their sample mean (average) approaches their theoretical mean, and that is the basis for Monte Carlo simulations and allows to build a stochastic model by the method of statistical trials.

Monte Carlo Methods or random sampling are used to run a simulation. For example, if it is sought to compute the time it will take to go from point A to point B, given some initial conditions, these conditions can be set at the start and the simulation can be run e.g. 1000 times to get an estimated time (the higher the number of runs or trials, the better the estimate).

Another application are simple integrals, which can be computed using the Riemann sum technique, a computationally expensive process in higher dimensions. Integration by Monte Carlo Methods can produce reasonably close approximations at a lower computational cost.

# Monte Carlo Integration

Calculating the integral of some expressions analytically is easy if the expression is simple. In many cases, however, analytical approaches to integration are not feasible, as the expression is very complicated, possibly without a closed analytical integral, or it can be evaluated at various points but it is not possible to know the corresponding equation. In such situations, Monte Carlo methods allow to generate an approximation of the integral, represented by the **basic Monte Carlo estimator** equation.

The average value of a function is given by

$f_{avg} = \dfrac{1}{b-a} \int_a^b f(x) dx$ 

Consequently, 

$\int_a^b f(x) dx = (b-a) f_{avg}$

$\left \langle F^N \right \rangle = (b-a) \dfrac{1}{N} \sum_{i=0}^{N-1} f(X_i) \quad$ where $N$ is the number of samples used in the approximation.

$\left \langle F^N \right \rangle = \bar{F}_N = E[F_N]$

$X_i = a + U(b - a) \quad$ where U is uniformly distributed between $0$ and $1$.

The law of large numbers states that as $N$ approaches infinity, the Monte Carlo approximation converges (in probability) to the right answer:

$P(\lim_{N \rightarrow \infty} E[F_N] = F) = 1$

 simply by evaluating the expression a large number of times at randomly selected points in the input space and counting the proportion that are less than the integrand at that point. The larger the number of simulations we run, the better the approximation.

The formula used above works only for a uniform PDF. The more generic formula is:

$\left \langle F^N \right \rangle = \dfrac{1}{N} \sum_{i=0}^{N-1} \dfrac{f(X_i)}{\text{PDF}(X_i)} \quad$ where $N$ is the number of samples used in the approximation.

Dividing $f(x)$ by $PDF(x)$ is necessary for non-constant PDFs . When samples aren't uniformly distributed, more samples are generated where the PDF is high and reversely, fewer samples are generated where the PDF is low. In a Monte Carlo integration though, the samples need to be uniformly distributed, in order not to be biased. Dividing $f(x)$ by $PDF(x)$ counterbalances this effect: when the PDF is high, dividing $f(x)$ by $PDF(x)$ will decrease the "weight" of these samples in the sum, compensating for the higher concentration of samples; when the PDF is low, fewer samples are generated, but dividing $f(x)$ by a low value increases the contribution of these samples.


```python
def monte_carlo_estimator(function, a, b, N=100000):
    points =  np.random.uniform(a, b, size=N)
    average = np.mean(list(map(function, points)))
    return (b - a) * average

n = 10000

f_of_x = lambda x: x**2
a = 2
b = 5
display(Markdown('$\int_{1}^{5} x^2 dx$'))
print(f'The integral Monte Carlo Estimator value is {monte_carlo_estimator(f_of_x, a, b, n):.4f} \n')

f_of_x = lambda x: np.sin(x)
a = 0
b = np.pi * 2
display(Markdown('$\int_{0}^{2\pi} sin(x) dx$'))
print(f'The integral Monte Carlo Estimator value is {monte_carlo_estimator(f_of_x, a, b, n):.4f} \n')

f_of_x = lambda x: np.exp(-x) / (1 + (x - 1)**2)
a = 0
b = 5
display(Markdown('$\int_{0}^{\infty} \dfrac{e^{-x}}{1+ (x-1)^2}dx$'))
print(f'The integral Monte Carlo Estimator value is {monte_carlo_estimator(f_of_x, a, b, n):.4f}')
```


$\int_{1}^{5} x^2 dx$


    The integral Monte Carlo Estimator value is 38.9021 
    
    


$\int_{0}^{2\pi} sin(x) dx$


    The integral Monte Carlo Estimator value is -0.0966 
    
    


$\int_{0}^{\infty} \dfrac{e^{-x}}{1+ (x-1)^2}dx$


    The integral Monte Carlo Estimator value is 0.6933
    

## Estimation of $\pi$

1. Generate random $(x, y)$ points belonging to a square of area $4 r^2$ centered at $(h,k)$, where $r$ is the side length. 
2. Calculate the points of the square that belong to a circle of ratio $r$ and area $\pi r^2$, that is, $\sqrt{(x-h)^2 + (y-k)^2} <= r^2$.
3. Calculate the approximation of $\pi$ as four times the proportion $p$ of points inside the circle over the total number of points:

$$p = \dfrac{\text{circle area}}{\text{square area}}$$

$$p = \dfrac{\pi r^2}{4 r^2} = \dfrac{\pi}{4}$$

$$\pi = 4 p$$


```python
num_points = 3000000
r = 1

square_x, square_y = np.random.uniform(-r, r, size=(2, num_points))
circle_x, circle_y = square_x[np.sqrt(square_x**2 + square_y**2) < r], square_y[np.sqrt(square_x**2 + square_y**2) < r]

pi = 4 * len(circle_x)/len(square_x)

plt.scatter(square_x, square_y, marker='.', s=1, color='darkred')
plt.scatter(circle_x, circle_y, marker='.', s=1, color='darkorange')
```


    
![image alt text]({static}../images/monte_carlo_methods_1.png)
    


## Custom Distribution Sampling

Most distributions can be sampled from their analytic expressions. However, when working with the operation of multiple distributions, this analytic formula can be complex. Monte Carlo Methods allow to sample direct from each distribution, and apply the operations to those generated values to obtain the new custom distribution.

Let $X \sim U(0,1) , \quad Y \sim Exp(0.5) , \quad Z \sim N(0,1) , \quad Q \sim B(2,5)$.

Let $A = (X + Y + Z) \cdot Q$.


```python
sample_size = 100000
A_distribution = []

X = np.random.uniform(0,1, size=sample_size)
Y = np.random.exponential(0.5, size=sample_size)
Z = np.random.normal(0, 1, size=sample_size)
Q = np.random.beta(2, 5, size=sample_size)
A =  (X + Y + 2*Z) * Q

plt.hist(A, bins=50, density=True)
```
![image alt text]({static}../images/monte_carlo_methods_2.png)
    

