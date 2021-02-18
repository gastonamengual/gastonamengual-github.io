Title: Simulated Annealing 
Date: 2020.09.25
Summary: Theoretical framework of the algorithm, including annealing schedules, and acceptance probability function. Implementation and application over benchmark functions. Evaluation with RMSE. 

<br>

**In collaboration with [Ezequiel L. Castaño](https://elc.github.io){: target="_blank"}.**

<hr>

# 1 Theoretical Framework

## 1.1 Introduction

Simulated annealing is a probabilistic technique for **approximating the global minimum** of a given function. It is often used when the search space is discrete, and for situations with a lot of local minima where algorithms like Gradient Descent would get stuck. The name of the algorithm comes from annealing in metallurgy, a technique involving **heating and controlled cooling of a material** to increase the size of its crystals and reduce their defects, both attributes depending on their thermodynamic free energy. Heating and cooling the material affects both the temperature and the thermodynamic free energy. 

When applied to engineering design, an **analogy is made between energy and the objective function**. The algorithm simulates a small random displacement of an atom that results in a change in energy. The design starts at an initial state of *high temperature*, where it has a high objective. Then, random perturbations are made to the initial state, generating *neighbor* states. If the objective of a new state is lower (the change in energy is negative), the energy state of the new configuration is lower, and the new configuration is accepted. If the objective is higher (the change in energy is positive), the new configuration has a higher energy state, but can still be accepted according to some acceptance probability function.

The **temperature progressively decreases** from an initial positive value to a certain threshold according to a certain *annealing schedule*, simulating the slow cooling in the annealing process, and influences the probability of acceptance of a *neighbor* state. When the temperature is higher, is it more likely that worse solutions are accepted, encouraging the **exploration** of the search space and allows the algorithm to more likely travel down a sub-optimal path to potentially find a global minimum. On the other hand, when the temperature is lower, the algorithm is less likely to accept worse solutions, promoting **exploitation**, that is, once the algorithm is in the right search space, there is no need to search other sections of the search space, and it should instead try to converge and find the global minimum.

The simulation can be performed by using the stochastic sampling method, and it is an adaptation of the **Metropolis–Hastings algorithm**.

## 1.2 Annealing Schedule 

The annealing procedure involves first "melting" the system at a high temperature, and then lowering the temperature by a constant factor $\alpha \; (0 < \alpha < 1$), taking enough steps at each temperature to keep the system close to equilibrium, until the system approaches the ground state. Analogously, the algorithm starts with an initial temperature $T_0$ set to a high value, and then it is decreased at each step $k$ following some *annealing* or *cooling schedule* until it reaches $0$ towards the end. 

There are several annealing schedules that can be considered.
    
### 1.2.1 Linear Cooling Schedule

$T(k) = T(k-1) - \alpha = T_0 - \alpha k$

### 1.2.2 Geometric Reduction Rule 

$T(k) = T(k-1) \cdot \alpha = T_0~\alpha^{k}$ 

### 1.2.3 Slow-Decrease Rule

$T(k) = \dfrac{T(k-1)}{1+\beta T(k-1)} = \dfrac{T_0}{\beta ~ k ~ T_0 + 1}$, where beta is an arbitrary constant

### 1.2.4 Exponential Cooling Schedule
$T(k) = T_0 \cdot e^{-\alpha k}$

### 1.2.5 Exponential Multiplicative Cooling Schedule
Proposed by Kirkpatrick, Gelatt and Vecchi (1983)

$T_k = T_0 \cdot \alpha^k \quad (0.8 < \alpha < 0.9)$

## 1.3 Acceptance Probability Function

The probability of making the transition from the current state $s$ to a neighbor state $s_{\text{new}}$ is specified by an acceptance probability function $P(e, e_{\text{new}}, T)$ that depends on the energies $e = E(s)$ and $e_{\text{new}} = E(s_{\text{new}})$ of the two states, and on the current temperature $T$. States with a smaller energy are better than those with a greater energy. $P$ must be positive even when $e_{\text{new}}$ is greater than $e$, to prevent the method from becoming stuck at a local minimum. When $T \rightarrow 0$, $P$ must tend to $0$ if $e_{\text{new}} > e = E(s)$, and to a positive value otherwise.

### 1.3.1 Boltzmann Distribution

The Boltzmann or Gibbs distribution is a probability distribution that gives the probability that a system will be in a certain state as a function of that state's energy and the temperature of the system. The distribution is expressed in the form:

$$p_i \propto e^{-\dfrac{\epsilon_i}{kT}}$$

where $p_i$ is the probability of the system being in state $i$, $\epsilon_i$ is the energy of that state, and a constant $k T$ of the distribution is the product of Boltzmann's constant $k$ and thermodynamic temperature $T$.

The present distribution will be considered as the **acceptance distribution**.

## 1.4 Algorithm Description

**1** Set an objective function $W(state)$, an initial state $s_0$, an initial temperature $T_0$, and an annealing schedule $AS$.
  
**2** Set a loop, the *temperature loop*. Inside it set a second loop, the *iteration loop*, which iterates $n$ iterations of Step 3, and then update the both the temperature, according to the $AS$, and the variance as explained in step 3  The *temperature loop* stops once the temperature has reached an end value. 

**3** Generate a neighbor state $s_{\text{new}}$ from a symmetric distribution (e.g normal distribution), with *variance* equal to the current temperature by the function domain. Calculate the absolute value of the difference between the objective value of the current state and the new neighbor state, that is,  $\Delta c = |W(s_0) - W(s_{\text{new}})|$. The probability of the new state being accepted is calculated as follows:

$$\left\{\begin{matrix}
1 & \text{if } S_0 \leq s_{\text{new}} \\ 
e^{-\Delta c / T(k)} & \text{otherwise}
\end{matrix}\right.$$

where $T(k)$ is the temperature in the current iteration of the *temperature loop*.

# 2 Implementation

## 2.1 Objective Functions


```python
def objective_function1(x):
    """Infinity77 - Univariate Problem 14"""
    return -np.exp(-x)*np.sin(2*pi*x)
bounds1 = [0, 4]
minimum1 = (0.2249, -0.7887)

def objective_function2(x):
    """Ackley"""
#     x = x - 1
    return -20 * np.exp(-0.2*x**2) - np.exp(np.cos(pi*2*x)) + np.exp(1) + 20 
bounds2 = [-32, 32]
minimum2 = (0, 0)

def objective_function3(x):
    "Gramacy & Lee"
    return np.sin(10*pi*x)/(2*x) + (x-1)**4
bounds3 = [-0.5, 2.5]
minimum3 = (0.1437791740738296, -2.8738989416296277) # Only Numerical Solution

def objective_function4(x):
    """Easom"""
    return np.cos(x)*np.exp(-((x-pi)**4))
bounds4 = [-100, 100]
minimum4 = (pi, -1)

def objective_function5(x):
    """Rastrigin"""
    return 10 + (x**2 - 10 * np.cos(2 * pi * x))
bounds5 = [-5.12, 5.12]
minimum5 = (0, 0)

def objective_function6(x):
    """Schwefel"""
    return -x * np.sin(np.sqrt(abs(x)))
bounds6 = [-500, 500]
minimum6 = (420.9687, -418.9829)

def objective_function7(x):
    """Styblinski-Tang"""
    return 0.5 * (x**4 - 16 * x**2 + 5 * x)
bounds7 = [-5, 5]
minimum7 = (-2.903534, -39.16599)

def objective_function8(x):
    """Infinity77 - Univariate Problem 15"""
    return (x**2 - 5*x + 6) / (x**2 + 1)

bounds8 = [-5, 5]
minimum8 = (1 + np.sqrt(2), 7 / 2 - 5 / np.sqrt(2))


objective_functions = [objective_function1, objective_function2, objective_function3, objective_function4, 
                       objective_function5, objective_function6, objective_function7, objective_function8]

bounds = [bounds1, bounds2, bounds3, bounds4, bounds5, bounds6, bounds7, bounds8]

minima = np.array([minimum1, minimum2, minimum3, minimum4, minimum5, minimum6, minimum7, minimum8])
```


![image alt text]({static}../images/simulated_annealing_1.png)
    


## 2.2 Simulated Annealing Algorithm


```python
def simulated_annealing(objective_function, bounds, initial_temperature=20, final_temperature=0.001, 
                        maximize=False, iterations=100, annealing_scheduler=slow_cooling, 
                        alpha=0.5, a=0.85, beta=0.30, steps=100):
    
    current_temperature = initial_temperature
    initial_solution = np.random.uniform(*bounds)
    
    best_solution = initial_solution   
    best_value = objective_function(best_solution)
    
    accepted_solutions = 0
    solutions_record = [best_solution]
    values_record = [best_value]
    temperature_record = [current_temperature]
    variance = current_temperature * (bounds[1] - bounds[0])
    
    for step in range(steps):
        
        for _ in range(iterations):

            neighbor = best_solution + np.random.normal(0, variance)
            neighbor = max(bounds[0], min(neighbor, bounds[1]))
            neighbor_value = objective_function(neighbor)

            delta_c = np.abs(best_value - neighbor_value)
            
            better = neighbor_value <= best_value
            
            if maximize:
                better = not better

            p = 1 if better else np.exp(-delta_c / current_temperature)

            if np.random.rand() > p:
                continue

            best_solution = neighbor
            best_value = neighbor_value
            accepted_solutions += 1
        
        current_temperature = annealing_scheduler(step, alpha=alpha, a=a, beta=beta, total_steps=steps,
                                                  initial_temperature=initial_temperature)
        variance = current_temperature * (bounds[1] - bounds[0])
        solutions_record.append(best_solution)
        values_record.append(best_value)
        temperature_record.append(current_temperature)
    
    return solutions_record, values_record, temperature_record, accepted_solutions/(iterations * len(temperature_record))
```

## 2.3 Annealing Schedules

### 2.3.1 Schedules Definition


```python
def linear_cooling(k, alpha, initial_temperature, total_steps=None, **kargs):
    "Linear"
    if total_steps:
        alpha = initial_temperature / total_steps
        return np.maximum(initial_temperature - alpha * k, 0)
    if isinstance(k, int):
        return np.maximum(initial_temperature - alpha * k, 0)
    return np.maximum(initial_temperature - alpha * k, np.zeros_like(len(k)))

def geometric_cooling(k, alpha, initial_temperature, **kargs):
    "Geometric"
    return initial_temperature * alpha**k

def slow_cooling(k, beta, initial_temperature, **kargs):
    "Slow"
    return initial_temperature / (beta * k * initial_temperature + 1)

def exponential_cooling(k, alpha, initial_temperature, **kargs):
    "Exponential"
    return initial_temperature * np.exp(-alpha * k)

def exponential_multiplicative_cooling(k, a, initial_temperature, **kargs):
    "Exponential Multiplicative"
    assert 0.8 < a < 0.9
    return initial_temperature * a**k

cooling_functions = [linear_cooling, geometric_cooling, slow_cooling,
                     exponential_cooling, exponential_multiplicative_cooling]

cooling_functions_names = np.array([function.__doc__ for function in cooling_functions])
```


```python
def cooling_schedules(initial_temperature=20, alpha=0.5, a=0.85, beta=0.05, steps=50):
    for cooling_function in cooling_functions:
        temperatures = cooling_function(xs, a=a, beta=beta,  total_steps=steps,
                                        alpha=alpha, initial_temperature=initial_temperature)
        plt.plot(xs, temperatures, label=cooling_function.__doc__)     
```


### 2.3.2 Schedules Comparison


```python
_, optimal_values = minima.T

cooling_function_rmse = []
for cooling_function in tqdm(cooling_functions):
    average_rmse = []
    for function, bound, optimal_value in zip(objective_functions, bounds, optimal_values):
        values = []
        for _ in range(100):
            _, value_record, *_ = simulated_annealing(function, bound, annealing_scheduler=cooling_function,
                                                      alpha=0.5, a=0.85, beta=0.15, steps=101)
            rmse = (np.array(value_record) - optimal_value) ** 2
            values.append(rmse)

        values_record = np.sqrt(np.mean(values, axis=0))
        average_rmse.append(values_record)
    
    avg_rmse = np.mean(average_rmse, axis=0)
    cooling_function_rmse.append(avg_rmse)
    plt.plot(avg_rmse[1:], label=cooling_function.__doc__)


```
    


    
![image alt text]({static}../images/simulated_annealing_2.png)
    



```python
average_rmse_overall = np.array(cooling_function_rmse).round(3)
average_rmse_overall_10 = average_rmse_overall[:, 10]
average_rmse_overall_20 = average_rmse_overall[:, 20]
average_rmse_overall_50 = average_rmse_overall[:, 50]
average_rmse_overall_100 = average_rmse_overall[:, 100]

data = np.array([cooling_functions_names, average_rmse_overall_10, average_rmse_overall_20, average_rmse_overall_50, average_rmse_overall_100,])

columns = ["Function", "Average RMSE (10)", "Average RMSE (20)", "Average RMSE (50)", "Average RMSE (100)"]
df = pd.DataFrame(data=data.T, columns=columns)
df
```




<div>
<table border="1">
  <thead>
    <tr>
      <th>Function</th>
      <th>Average RMSE (10)</th>
      <th>Average RMSE (20)</th>
      <th>Average RMSE (50)</th>
      <th>Average RMSE (100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear</td>
      <td>25.95</td>
      <td>15.802</td>
      <td>10.831</td>
      <td>0.959</td>
    </tr>
    <tr>
      <td>Geometric</td>
      <td>0.179</td>
      <td>0.066</td>
      <td>0.066</td>
      <td>0.066</td>
    </tr>
    <tr>
      <td>Slow</td>
      <td>1.258</td>
      <td>0.526</td>
      <td>0.188</td>
      <td>0.058</td>
    </tr>
    <tr>
      <td>Exponential</td>
      <td>0.919</td>
      <td>0.045</td>
      <td>0.043</td>
      <td>0.043</td>
    </tr>
    <tr>
      <td>Exponential Multiplicative</td>
      <td>13.672</td>
      <td>1.696</td>
      <td>0.017</td>
      <td>0.011</td>
    </tr>
  </tbody>
</table>
</div>



# 3 Optimization Execution


```python
results = []

for objective_function, bound in zip(objective_functions, bounds):
    best_solutions = []
    best_values = []
    
    for _ in tqdm(range(500)):
        solutions_record, values_record, temperature_record, accepted_proportions = simulated_annealing(objective_function, bound)
        best_solutions.append(solutions_record[-1])
        best_values.append(values_record[-1])
    
    result = [np.mean(best_solutions), np.mean(best_values), np.mean(accepted_proportions)]
    results.append(result)

results = np.array(results)
```

# 4 Results


```python
best_solution, best_value, acceptance_rate = results.T
optimal_solution, optimal_value = minima.T

columns = ['Solution Abs Error', 'Solution Rel Error', 'Value Abs Error', 'Value Rel Error',
           'Optimal Solution', 'Optimal Value', 'Acceptance Rate']

function_names = np.array([function.__doc__ for function in objective_functions])

data = np.array([best_solution - optimal_solution, (best_solution / optimal_solution - 1) * 100,
                 best_value - optimal_value, (best_value / optimal_value - 1) * 100,
                 optimal_solution, optimal_value, acceptance_rate*100])

df = pd.DataFrame(data=data.T, columns=columns)
df = df.round(4)
df['Function'] = function_names.T
```




<div>
<table border="1">
  <thead>
    <tr>
      <th>Solution Abs Error</th>
      <th>Solution Rel Error</th>
      <th>Value Abs Error</th>
      <th>Value Rel Error</th>
      <th>Optimal Solution</th>
      <th>Optimal Value</th>
      <th>Acceptance Rate</th>
      <th>Function</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>-0.0003</td>
      <td>-0.1276</td>
      <td>0.0049</td>
      <td>-0.6231</td>
      <td>0.2249</td>
      <td>-0.7887</td>
      <td>62.5254</td>
      <td>Problem 14</td>
    </tr>
    <tr>
      <td>0.0002</td>
      <td>0.0193</td>
      <td>0.0061</td>
      <td>0.6137</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>16.2463</td>
      <td>Ackley</td>
    </tr>
    <tr>
      <td>0.0001</td>
      <td>0.0492</td>
      <td>0.0052</td>
      <td>-0.1810</td>
      <td>0.1438</td>
      <td>-2.8739</td>
      <td>25.1149</td>
      <td>Gramacy &amp; Lee</td>
    </tr>
    <tr>
      <td>-0.0028</td>
      <td>-0.0883</td>
      <td>0.0050</td>
      <td>-0.4995</td>
      <td>3.1416</td>
      <td>-1.0000</td>
      <td>57.6597</td>
      <td>Easom</td>
    </tr>
    <tr>
      <td>-0.0001</td>
      <td>-0.0132</td>
      <td>0.0054</td>
      <td>0.5410</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>8.6075</td>
      <td>Rastrigin</td>
    </tr>
    <tr>
      <td>0.0047</td>
      <td>0.0011</td>
      <td>0.0051</td>
      <td>-0.0012</td>
      <td>420.9687</td>
      <td>-418.9829</td>
      <td>0.7030</td>
      <td>Schwefel</td>
    </tr>
    <tr>
      <td>0.0002</td>
      <td>-0.0075</td>
      <td>0.0058</td>
      <td>-0.0148</td>
      <td>-2.9035</td>
      <td>-39.1660</td>
      <td>5.8015</td>
      <td>Styblinski-Tang</td>
    </tr>
  </tbody>
</table>
</div>

<hr>

# References

https://en.wikipedia.org/wiki/Simulated_annealing

https://www.youtube.com/watch?v=T28fr9wDZrg&t=277s&ab_channel=SolvingOptimizationProblems

https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7

http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/

Problem 14: http://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem14

Ackley: http://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Ackley

Gramacy & Lee: http://benchmarkfcns.xyz/benchmarkfcns/gramacyleefcn.html

Easom: http://www.geatbx.com/ver_3_3/fcneaso.html#:~:text=The%20Easom%20function%20%5BEas90%5D%20is,2%2Dpi)%5E2))%3B

Rastrigin: https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/rastrigin.html

Schwefel: http://www.geatbx.com/ver_3_3/fcnfun7.html

Styblinski-Tang: https://www.sfu.ca/~ssurjano/stybtang.html

Problem 15: http://infinity77.net/global_optimization/test_functions_1d.html#go_benchmark.Problem15
