Title: Gradient Descent
Date: 2020.12.29
Summary: Explanation and implementation of Gradient Descent. Application on benchmark functions. Evaluation with RMSE. 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
          'axes.prop_cycle': plt.cycler(color=["darkcyan", "saddlebrown", "mediumpurple", "olivedrab", "darkseagreen", "darkkhaki", "darkgoldenrod", "deepskyblue", "firebrick", "palevioletred"]),}
plt.rcParams.update(config)
```

# Definition

Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.

It is based on the observation that if the multivariate function $F(X)$ is defined and differentiable in a neighborhood of a point $a$, then $F(X)$ decreases fastest if one goes from $a$ in the direction of the negative gradient of $F$ at $a$, $- \nabla F(a)$. It follows that if $a_{n+1} = a_n - \gamma \nabla F(a_n)$ for $\gamma \in \mathbb{R}^2$ small enough, then $F(a_n) \geq F(a_{n+1})$.

$\gamma \nabla F(a)$ is subtracted from $a$ because it is sought to move against the gradient toward the local minimum. An initial guess x_0 is made for a local minimum of $F$, and the sequence $x_0, x_1, x_2, \dotsc$ is considered such that

$$x_{n+1} = x_n - \gamma_n \nabla F(x_n), \; n \geq 0$$

The monotonic sequence $F(x_0) \geq F(x_1) \geq F(x_2) \geq \dotsc$ may converge to the desired local minimum.


```python
def gradient_descent(gradient, initial, learning_rate=0.01, num_epochs=5000):
    x = initial
    points = []
    for i in range(num_epochs):
        x = x - learning_rate * gradient(x)
        points.append(x)
    
    return x, num_epochs, np.array(points)
```

## Testing on Benchmark Functions

Gradient descent optimization will be applied to 6 benchmark functions, to test whether it approximates the different minima.


```python
def ackley(points):
    '''Ackley'''
    x, y = points
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))

def ackley_gradient(point):
    x, y = point
    dx = (2.82842 * np.exp(-0.2 * np.sqrt(0.5) * np.sqrt(x**2 + y**2)) * x) / np.sqrt(x**2 + y**2)
    dy = (2.82842 * np.exp(-0.2 * np.sqrt(0.5) * np.sqrt(x**2 + y**2)) * y) / np.sqrt(x**2 + y**2)
    return np.array([dx, dy])

ackley_minimum = [(0, 0)]
ackley_xbounds = (-4, 4)
ackley_ybounds = (-4, 4)

######################################

def rosenbrock(point):
    '''Rosenbrock'''
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(point):
    x, y = point
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    
    return np.array([dx, dy])
rosenbrock_minimum = [(1, 1)]
rosenbrock_xbounds = (0, 1.5)
rosenbrock_ybounds = (0, 1.5)

######################################

def himmelblau(point):
    '''Himmelblau'''
    x, y = point
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def himmelblau_gradient(point):
    x, y = point
    dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
    dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
    return np.array([dx, dy])

himmelblau_minima = [(3,2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]
himmelblau_xbounds = (-6, 6)
himmelblau_ybounds = (-6, 6)

######################################

def easom(point):
    '''Easom'''
    x, y = point
    return -np.cos(x) * np.cos(y) * np.exp(-(x - np.pi)**2 -(y - np.pi)**2)

def easom_gradient(point):
    x, y = point
    dx = - np.cos(y) * (np.exp(-x**2 + 2 * np.pi * x - 2 * np.pi**2 - y**2 + 2 * np.pi * y) * np.cos(x) * (-2*x + 2 * np.pi) - np.exp(-x**2 + 2 * np.pi * x - 2 * np.pi**2 - y**2 + 2 * np.pi * y) * np.sin(x))
    dy = - np.cos(x) * (np.exp(-x**2 + 2 * np.pi * x - 2 * np.pi**2 - y**2 + 2 * np.pi * y) * np.cos(y) * (-2*y + 2 * np.pi) - np.exp(-x**2 + 2 * np.pi * x - 2 * np.pi**2 - y**2 + 2 * np.pi * y) * np.sin(y))
    return np.array([dx, dy])

easom_minimum = [(np.pi, np.pi)]
easom_xbounds = (-100, 100)
easom_ybounds = (-100, 100)

######################################

def rastrigin(point):
    '''Rastrigin'''
    x, y = point
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

def rastrigin_gradient(point):
    x, y = point
    dx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    dy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)
    return np.array([dx, dy])

rastrigin_minimum = [(0, 0)]
rastrigin_xbounds = (-1, 1)
rastrigin_ybounds = (-1, 1)

######################################

def griewank(point):
    '''Griewank'''
    x, y = point
    return 1 + x**2 / 4000 + y**2 / 4000 - np.cos(x) * np.cos(y / 2 * np.sqrt(2))

def griewank_gradient(point):
    x, y = point
    dx = np.sin(x) * np.cos(y/np.sqrt(2)) + 1 / 4000
    dy = (np.cos(x) * np.sin(y / np.sqrt(2))) / np.sqrt(2) + 1 / 4000
    return np.array([dx, dy])

griewank_minimum = [(0, 0)]
griewank_xbounds = (-30, 30)
griewank_ybounds = (-30, 30)

######################################

objective_functions = [ackley, rosenbrock, himmelblau, easom, rastrigin, griewank]
gradients = [ackley_gradient, rosenbrock_gradient, himmelblau_gradient, easom_gradient, rastrigin_gradient, griewank_gradient]
minima = np.array([ackley_minimum, rosenbrock_minimum, himmelblau_minima, easom_minimum, rastrigin_minimum, griewank_minimum], dtype=object)
x_bounds = [ackley_xbounds, rosenbrock_xbounds, himmelblau_xbounds, easom_xbounds, rastrigin_xbounds, griewank_xbounds]
y_bounds = [ackley_ybounds, rosenbrock_ybounds, himmelblau_ybounds, easom_ybounds, rastrigin_ybounds, griewank_ybounds]
```


```python
num_replicates = 500
learning_rates = [0.02, 0.002, 0.01, 0.2, 0.002, 0.02]
epochs_numbers = [500, 3000, 20, 1500, 1500, 1500]
levels = [20, 60, 100, 15, 30, 40]
cmaps = ['viridis', 'cividis', 'Spectral', 'magma', 'twilight_shifted', 'twilight']

fig, axes = plt.subplots(3, 2, figsize=(16,14))

for ax, function, gradient, minimum, x_bound, y_bound, learning_rate, epoch_number, level, cmap in tqdm(zip(axes.reshape(-1), objective_functions, gradients, minima, x_bounds, y_bounds, learning_rates, epochs_numbers, levels, cmaps)):
    
    # Plot contour line
    x = np.linspace(x_bound[0], x_bound[1], 1000)
    y = np.linspace(y_bound[0], y_bound[1], 1000)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    fig = ax.contourf(X, Y, Z, level, cmap=cmap, zorder=1)
    plt.colorbar(fig, ax = ax)
    solutions = []
    
    for i in range(num_replicates):
        
        # Initial point
        initial = (np.random.uniform(x_bound[0], x_bound[1]), np.random.uniform(y_bound[0], y_bound[1]))
        
        # Gradient descent
        optimal_solution, final_iteration, points = gradient_descent(gradient, initial, learning_rate=learning_rate, num_epochs=epoch_number)
        solutions.append(optimal_solution)
        
        if i > 200:
            continue
        
        points_x = points[:,0]
        points_y = points[:,1]
        
        # Plot first point
        ax.scatter(points_x[0], points_y[0], marker='D', color='darkslategray', s=50, zorder=2)
        # Plot line between points
        ax.plot(points_x[:-1], points_y[:-1], color='whitesmoke', linewidth=.7, zorder=3)
        # Plot last point
        ax.scatter(points_x[-1], points_y[-1], marker='+', color='fuchsia', s=100, zorder=4)
    
    solutions = np.array(solutions)
    
    # Calculate error
    rmse = np.sqrt(np.mean(((function(solutions.T) - function(minimum[0]))**2)))
    
    # Title and lims
    ax.set_title(f'{function.__doc__} - {epoch_number} epochs - RMSE = {rmse:.3g}', fontsize=17)
    ax.set_xlim(x_bound[0], x_bound[1])
    ax.set_ylim(y_bound[0], y_bound[1])
    
    # Plot actual minima
    for m in minimum:
        ax.scatter(m[0], m[1], color='#FDFDFD', marker='x', linewidths=7, s=200, zorder=5, edgecolors='#333333')
    ax.text(x_bound[0] + (x_bound[1] - x_bound[0])*0.05, y_bound[0] + (y_bound[1] - y_bound[0])*0.06, f'$Min f(x, y) = {function(minimum[0]):.2g}$', bbox=dict(facecolor='white', alpha=1), fontsize=13, zorder=6)

plt.suptitle(f'Gradient Descent on Benchmark Functions (Fixed learning rate - {num_replicates} replicates)', y=1, fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.savefig(f'gradient_descent_benchmark_functions_{num_replicates}.png', dpi=400, pad_inches=0.2, bbox_inches='tight')
plt.show()
```



    
![image alt text]({static}../images/gradient_descent.png)
    


It can be observed that Gradient Descent manages to get a fairly precise approximation of the global minima, but gets frequently stuck when many local minima surround the global minimum, as in the Easom and Griewank functions. 

The gradient descent algorithm explained is the basic and simplest version. There are other versions such as stochastic gradient descent, mini-batch gradient descent, and techniques such as regularization and adaptive learning rate attempt to improve the performance of the algorithm.  

# References

Gradient Descent: <a href="https://en.wikipedia.org/wiki/Gradient_descent">https://en.wikipedia.org/wiki/Gradient_descent</a>

Ackley Function: <a href="https://en.wikipedia.org/wiki/Ackley_function">https://en.wikipedia.org/wiki/Ackley_function</a>

Rosenbrock Function: <a href="https://en.wikipedia.org/wiki/Rosenbrock_function">https://en.wikipedia.org/wiki/Rosenbrock_function</a>
 
Himmelblau Function: <a href="https://en.wikipedia.org/wiki/Himmelblau's_function">https://en.wikipedia.org/wiki/Himmelblau's_function</a>
 
Easom Function: <a href="https://www.sfu.ca/~ssurjano/easom.html">https://www.sfu.ca/~ssurjano/easom.html</a>
 
Rastrigin Function: <a href="https://en.wikipedia.org/wiki/Rastrigin_function">https://en.wikipedia.org/wiki/Rastrigin_function</a>
 
Griewank Function: <a href="https://en.wikipedia.org/wiki/Griewank_function">https://en.wikipedia.org/wiki/Griewank_function</a>
