Title: Canonical Genetic Algorithm
Date: 2020.09.10
Summary: Application of Canonical Genetic Algorithm in a function maximization problem.

**Notebook written by Gastón Amengual.**

<hr>

### Maximize

$f(x) = \left (\dfrac{x}{2^{30} -1} \right)^2 \quad x \in [0, 2^{30}-1]$


```python
def objective_function(x):
    return (x / (2 ** 30 - 1)) ** 2
```


```python
x = np.linspace(0,2**30-1, 100)
plt.plot(x,objective_function(x), label='Maximum Value: 1\nMaximum x = 1073741823')
```


    
![image alt text]({static}../images/canonical_genetic_algorithm_1.png)
    



```python
num_chromosomes = 10
num_genes = 30
num_generations = 150
crossover_probability = 0.9
mutation_probability = 0.05
```


```python
population = []
fitness = []

optimal_value = [0]
min_values = [0]
mean_values = [0]
optimal_solutions = [0]

# Initial Population
for i in range (num_chromosomes):
    chromosome = np.random.randint(0, 2, size=num_genes)
    chromosome = (np.array2string(chromosome, separator='')[1:-1])
    population.append(chromosome)

for i in range(num_generations):
    sum_function_values = np.sum([objective_function(int(chromosome,2)) for chromosome in population])
    fitness = [objective_function(int(chromosome,2)) / sum_function_values for chromosome in population]
    
    parents = np.random.choice(population, size=num_chromosomes, p=fitness)
    population = []
    
    # Crossover
    while parents.size != 0:
        parent1 = parents[-1]
        parent2 = parents[-2]        
        
        if crossover_probability > np.random.rand():
            cut = np.random.randint(num_genes, size=2)
            child1 = parent1[:cut[0]] + parent2[cut[0]:]          
            child2 = parent2[:cut[1]] + parent1[cut[1]:]
        else:
            child1 = parent1
            child2 = parent2
            
        population.append(child1)
        population.append(child2)
        parents = np.delete(parents, np.s_[-2:])
    
    # Mutation
    for i in range(num_chromosomes):
        if mutation_probability > np.random.rand():
            mutation_index = np.random.randint(num_genes)
            aux = list(population[i])
            aux[mutation_index] = '0' if aux[mutation_index] == 1 else '1'
            population[i] = ''.join(aux)

    obj_function_value = np.array([objective_function(int(chromosome,2)) for chromosome in population])
    chromosome_decimal_value = np.array([int(chromosome,2) for chromosome in population])

    optimal_value.append(obj_function_value.max())
    min_values.append(obj_function_value.min())
    mean_values.append(obj_function_value.mean())
    optimal_solutions.append(chromosome_decimal_value.max())
```


    Optimal solution value accuracy: -1e+02%
    Optimal value accuracy: 0%
    


    
![image alt text]({static}../images/canonical_genetic_algorithm_2.png)
    

