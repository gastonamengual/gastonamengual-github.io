Title: Roulette Simulation: Statistical Analysis
Date: 2020.04.15
Summary: Simulation of a Casino Roulette, for which the mean and variance are analyzed, and empirical demonstration of the Law of Great Numbers.

## <a target="blank" href="{static}../documents/roulette_simulation_statistical_analysis_paper.pdf">Read the paper here!</a>

<hr>

First, the roulette size and the analytical (henceforth *theoretical*) or population parameters are calculated.


```python
roulette_size = 37

theoretical_mean = np.mean(np.arange(0, roulette_size))
theoretical_var = np.var(np.arange(0, roulette_size))
theoretical_std = np.std(np.arange(0, roulette_size))
theoretical_frequency = 1 / roulette_size
```

## Roulette


```python
def roulette(throws):
    
    means = []
    variances = []
    stds = []
    frequencies = []
    numbers_thrown = []
    
    number_of_frequency = np.random.randint(0, roulette_size)
    frequency_count = 0
    
    for i in range(1, throws + 1):

        number_thrown = np.random.randint(0, roulette_size)

        if number_thrown == number_of_frequency:
            frequency_count += 1

        frequencies.append(frequency_count / i)
            
        numbers_thrown.append(number_thrown)

        means.append(np.mean(numbers_thrown))
        stds.append(np.std(numbers_thrown))
        variances.append(np.var(numbers_thrown))

    return ({'means': np.array(means), 
             'variances': np.array(variances), 
             'stds': np.array(stds), 
             'frequencies': np.array(frequencies)})
```

## Simulation Runs


```python
# The experiment consists of 30 runs
runs = 30

# Runs for 100 roulette throws

means_total_100 = []
stds_total_100 = []
variances_total_100 = []
frequencies_total_100 = []

for i in tqdm(range(runs)):
    statistics = roulette(100)    

    means_total_100.append(statistics['means'])
    stds_total_100.append(statistics['variances'])
    variances_total_100.append(statistics['stds'])
    frequencies_total_100.append(statistics['frequencies'])

# Runs for 15000 roulette throws
    
means_total_15000 = []
stds_total_15000 = []
variances_total_15000 = []
frequencies_total_15000 = []

for i in tqdm(range(runs)):
    statistics = roulette(15000)    

    means_total_15000.append(statistics['means'])
    stds_total_15000.append(statistics['variances'])
    variances_total_15000.append(statistics['stds'])
    frequencies_total_15000.append(statistics['frequencies'])
```
    

## Visualization

![image alt text]({static}../images/roulette_simulation_statistical_analysis_1.png)


# 15000 throws

![image alt text]({static}../images/roulette_simulation_statistical_analysis_2.png)