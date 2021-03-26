Title: Roulette Simulation: Strategies Analysis
Date: 2020.05.01
Summary: Simulation of Martingale, d'Alembert and All-in Strategies in a Casino Roulette, for which the capital flow is analyzed.

## <a href="{static}../documents/roulette_simulation_strategies_analysis_paper.pdf">Read the paper here!</a>

<hr>

# Strategies


```python
def roulette_strategies(initial_cash, initial_bet_amount, kind, iterations=100, infinite=False):
        
    if infinite == True:
        threshold = -np.inf
    else:
        threshold = 0

    current_cash = initial_cash
    current_bet_amount = initial_bet_amount
    
    bet_results = []
    cash_flow = [current_cash]

    for i in range(iterations):
        
        if current_cash - current_bet_amount >= threshold:
            number_thrown = np.random.randint(0, roulette_size, )
            
            if number_thrown in roulette:
                if number_thrown != 0:
                    # Win scenario
                    current_cash = current_cash + current_bet_amount
                    bet_results.append(1)

                    if kind == 'martingale': 
                        current_bet_amount = initial_bet_amount
                    if kind == 'dalembert':
                        if current_bet_amount > initial_bet_amount:
                            current_bet_amount -= 1
                    if kind == 'all_in':
                        current_bet_amount = current_cash
                else:
                    bet_results.append(np.nan)
                    
            
            # Loss scenario
            else:
                current_cash = current_cash - current_bet_amount
                bet_results.append(0)
                
                if kind == 'martingale':
                    current_bet_amount = current_bet_amount * 2
                if kind == 'dalembert':
                    current_bet_amount += 1
                if kind == 'all_in':
                    pass
                
            cash_flow.append(current_cash)
        
        else: 
            cash_flow.append(np.nan)
            bet_results.append(np.nan)
            
    return np.array(cash_flow), np.array(bet_results)
```

# Simulation Runs


```python
roulette = [0, 2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35] # considers only blue and zero
roulette_size = 37

initial_bet_amount = 100
initial_cash = 1000

runs = 100
iterations = 100
```


```python
# Martingale
martingale_cash_list = []
martingale_bets_list = []
for i in range(runs):
    cash_flow, bet_results = roulette_strategies(initial_cash, initial_bet_amount, kind='martingale', iterations=iterations, infinite=False)
    martingale_cash_list.append(cash_flow)
    martingale_bets_list.append(bet_results)
    
# Martingale Infinite
martingale_infinite_cash_list = []
martingale_infinite_bets_list = []
for i in range(runs):
    cash_flow, bet_results = roulette_strategies(initial_cash, initial_bet_amount, kind='martingale', iterations=iterations, infinite=True)
    martingale_infinite_cash_list.append(cash_flow)
    martingale_infinite_bets_list.append(bet_results)

# Dalembert
dalembert_cash_list = []
dalembert_bets_list = []
for i in range(runs):
    cash_flow, bet_results = roulette_strategies(initial_cash, initial_bet_amount, kind='dalembert', iterations=iterations)
    dalembert_cash_list.append(cash_flow)
    dalembert_bets_list.append(bet_results)

# All_in
all_in_cash_list = []
all_in_bets_list = []
for i in range(runs):
    cash_flow, bet_results = roulette_strategies(initial_cash, initial_bet_amount, kind='all_in', iterations=iterations)
    all_in_cash_list.append(cash_flow)
    all_in_bets_list.append(bet_results)
```

# Visualization

## Cash Flow Plots

Shows the cash flow for all roulette throws in all 100 iterations

![image alt text]({static}../images/roulette_simulation_strategies_analysis_1.png)    


## Wins Relative Frequency Plots

Calculates relative frequency of wins for each strategy


```python
martingale_rel_freq = []
for i in range(iterations):
    count = 0
    for j in range(iterations):
        if martingale_bets_list[j][i] == 1:
            count += 1
    martingale_rel_freq.append(count / iterations)
    
martingale_infinite_rel_freq = []
for i in range(iterations):
    count = 0
    for j in range(iterations):
        if martingale_infinite_bets_list[j][i] == 1:
            count += 1
    martingale_infinite_rel_freq.append(count / iterations)
    
dalembert_rel_freq = []
for i in range(iterations):
    count = 0
    for j in range(iterations):
        if dalembert_bets_list[j][i] == 1:
            count += 1
    dalembert_rel_freq.append(count / iterations)

all_in_rel_freq = []
for i in range(iterations):
    count = 0
    for j in range(iterations):
        if all_in_bets_list[j][i] == 1:
            count += 1
    all_in_rel_freq.append(count / iterations)
```


![image alt text]({static}../images/roulette_simulation_strategies_analysis_2.png)



## Win Percentage for 100 Iterations

Times the final capital was greater than initial capital at final roulette throw in all 100 iterations


```python
wins = 0
for run in range(runs):
    if(martingale_cash_list[run][iterations] > initial_cash):
        wins += 1
martingale_wins = wins/runs

wins = 0
for run in range(runs):
    if(martingale_infinite_cash_list[run][iterations] > initial_cash):
        wins += 1
martingale_infinite_wins = wins/runs

wins = 0
for run in range(runs):
    if(dalembert_cash_list[run][iterations] > initial_cash):
        wins += 1
dalembert_wins = wins/runs

wins = 0
for run in range(runs):
    if(all_in_cash_list[run][iterations] > initial_cash):
        wins += 1
all_in_wins = wins/runs
```

After 100 iterations

Martingale wins percentage: 11%

Martingale infinite wins percentage: 99%

Dalembert wins percentage: 52%

All in wins percentage: 0%
    

## Final Capital for 100 Iterations (12 for All-in)


```python
final_amount = []
for run in range(runs):
    if(martingale_cash_list[run][iterations] > initial_cash):
        final_amount.append(martingale_cash_list[run][iterations])
martingale_final_capital = np.mean(final_amount)

final_amount = []
for run in range(runs):
    if(martingale_infinite_cash_list[run][iterations] > initial_cash):
        final_amount.append(martingale_infinite_cash_list[run][iterations])
martingale_infinite_final_capital = np.mean(final_amount)

final_amount = []
for run in range(runs):
    if(dalembert_cash_list[run][iterations] > initial_cash):
        final_amount.append(dalembert_cash_list[run][iterations])
dalembert_final_capital = np.mean(final_amount)

final_amount = []
for run in range(runs):
    if(all_in_cash_list[run][12] > initial_cash):
        final_amount.append(all_in_cash_list[run][12])
all_in_final_capital = np.mean(final_amount)

```

Martingale final capital mean: $5972.73

Martingale infinite final capital mean: $5607.07

Dalembert final capital mean: $1758.40

All in final capital mean: $204800.00
    

## Maximum Capital Obtained

Calculates the maximum capital obtained in all iterations of all runs.


```python
martingale_max_capital = 0
for run in range(runs):
    for i in range(iterations):
        if(martingale_cash_list[run][i] > martingale_max_capital):
            martingale_max_capital = martingale_cash_list[run][i]

martingale_infinite_max_capital = 0
for run in range(runs):
    for i in range(iterations):
        if(martingale_infinite_cash_list[run][i] > martingale_infinite_max_capital):
            martingale_infinite_max_capital = martingale_infinite_cash_list[run][i]

dalembert_max_capital = 0
for run in range(runs):
    for i in range(iterations):
        if(dalembert_cash_list[run][i] > dalembert_max_capital):
            dalembert_max_capital = dalembert_cash_list[run][i]

all_in_max_capital = 0
for run in range(runs):
    for i in range(iterations):
        if(all_in_cash_list[run][i] > all_in_max_capital):
            all_in_max_capital = all_in_cash_list[run][i]
```

Martingale max capital: $6500.00

Martingale infinite max capital: $6900.00

Dalembert max capital: $2936.00

All in max capital: $563200.00
    

## Probabilistic Martingale Verification


```python
mean = np.nanmean(np.array(martingale_cash_list), axis=0)
mean_difference = np.diff(mean,n=1)
mean_difference_percentage = np.divide(mean_difference, np.mean(mean))

x = np.arange(0, iterations)
plt.figure(figsize=(16, 4))
plt.plot(x, mean_difference_percentage)
plt.axhline(0, color='black', linestyle='--', linewidth=3)

plt.xlabel('Iterations')
plt.ylabel('Diference')
plt.title('Scaled Difference Between Each Mean and its Previous Mean')

plt.tight_layout()
plt.savefig('mean_difference.png', dpi=300)
plt.show()

display(Markdown('**Credibility Intervals**'))
errors = [0.0001, 0.001, 0.01, 0.05, 0.10, 0.15]
for error in errors:
    probability = np.mean(abs(mean_difference_percentage) < error)    
    display(Markdown("$P(|E(X_{n+1} - X_n)| <" + f"{error}) = $" + f" {probability}"))
```


![image alt text]({static}../images/roulette_simulation_strategies_analysis_3.png)
    



**Credibility Intervals**



$P(|E(X_{n+1} - X_n)| <0.0001) = 0.01$



$P(|E(X_{n+1} - X_n)| <0.001) = 0.02$



$P(|E(X_{n+1} - X_n)| <0.01) = 0.31$



$P(|E(X_{n+1} - X_n)| <0.05) = 0.88$



$P(|E(X_{n+1} - X_n)| <0.1) = 0.99$



$P(|E(X_{n+1} - X_n)| < 0.15) = 1$