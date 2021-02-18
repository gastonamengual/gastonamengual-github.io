Title: M/M/1 Queue Simulation
Date: 2020.06.24
Summary: Simulation of a MM1 Queue and analysis of average delay in queue, average customers in queue, server busy proportion, and total time it takes for n users to get service. 

## <a href="{static}../documents/mm1_queue_simulation_paper.pdf">Read the paper here!</a>


**Notebook written by Gastón Amengual as a part of the Simulation Course at the National Technology University (UTN).**

<hr>

## MM1 Queue Simulation


```python
def mm1_queue_simulation(total_customers_served, customer_rate=1, service_rate=2):
    
    busy = 1
    idle = 0
    
    # INITIALIZE
    current_event_time = 0
    server_status = idle
    customers_in_queue = 0
    time_last_event = 0
    customers_out_queue = 0
    total_wait = 0
    area_of_q = 0
    area_of_b = 0
    time_arrival = np.array([]),
    events = {'arrival': 0 + np.random.exponential(1 / customer_rate),
              'departure': np.inf}
    
    # RUN WHILE TOTAL CUSTOMERS ARE NOT SERVED AND THERE ARE CUSTOMERS IN QUEUE AND THERE IS A CUSTOMER BEING SERVED
    while not((customers_out_queue == total_customers_served) and (customers_in_queue == 0) and (events['departure'] == np.inf)):

        # SET EVENT TYPE AND TIME
        current_event_type, current_event_time = min(events.items(), key=lambda x: x[1])

        time_since_last_event = current_event_time - time_last_event
        time_last_event = current_event_time
        area_of_q += customers_in_queue * time_since_last_event
        area_of_b += server_status * time_since_last_event

        # ARRIVAL EVENT
        if current_event_type == 'arrival':

            # Check total customers served
            if customers_in_queue + customers_out_queue != total_customers_served:

                # Set new arrival
                events['arrival'] = current_event_time + np.random.exponential(1/customer_rate)

                # If server is busy, customer enters queue
                if server_status == busy:
                    customers_in_queue += 1
                    time_arrival = np.append(time_arrival,current_event_time)

                # If server is idle, customer gets service
                else:
                    customers_out_queue += 1
                    server_status = busy

                    # Set customer departure
                    events['departure'] = current_event_time + np.random.exponential(1/service_rate)

            # Discards arrival event
            else:
                events['arrival'] = np.inf

        # DEPARTURE EVENT
        elif current_event_type == 'departure':

            # If there is no queue, server is idle
            if customers_in_queue == 0:
                server_status = idle
                events['departure'] = np.inf

            # If there is queue, serve next customer
            else:

                # Decrement queue
                customers_in_queue -= 1

                # Calculates waiting time
                if time_arrival.size != 0:
                    wait = current_event_time - time_arrival[0]
                else:
                    wait = current_event_time

                total_wait += wait
                customers_out_queue += 1

                # Set customer departure
                events['departure'] = current_event_time + np.random.exponential(1/service_rate)

                #Move customers in queue (if any) up one place
                time_arrival = np.roll(time_arrival,(customers_in_queue),-1)
                time_arrival = time_arrival[:customers_in_queue]

    # Calculate performance measures
    average_delay_in_queue = total_wait / customers_out_queue
    average_customers_in_queue = area_of_q / current_event_time
    server_busy_proportion = area_of_b / current_event_time
    
    return average_delay_in_queue, average_customers_in_queue, server_busy_proportion, current_event_time
```

## Simulation Runs


```python
customer_rate = 1
service_rate = 2
size = 500
total_customers_served_list = [15, 500, 1000]

stats_list = []

for total_customers_served in tqdm(total_customers_served_list):
    stats = np.array([mm1_queue_simulation(total_customers_served) for i in range(size)])
    stats_list.append(stats)
```


$\lambda_c = 1 \quad \lambda_s = 2$



### 15 customers served



$E[\hat d(n)] = 0.332 \quad Std[\hat d(n)] = 0.324$



$E[\hat q(n)] = 0.341 \quad Std[\hat q(n)] = 0.356$



$E[\hat u(n)] = 0.485 \quad Std[\hat u(n)] = 0.146$



$E[T(n)] = 16.04 \quad Std[T(n)] = 3.59$



### 500 customers served



$E[\hat d(n)] = 0.48 \quad Std[\hat d(n)] = 0.117$



$E[\hat q(n)] = 0.481 \quad Std[\hat q(n)] = 0.128$



$E[\hat u(n)] = 0.497 \quad Std[\hat u(n)] = 0.031$



$E[T(n)] = 502.4 \quad Std[T(n)] = 23$



### 1000 customers served



$E[\hat d(n)] = 0.495 \quad Std[\hat d(n)] = 0.0829$



$E[\hat q(n)] = 0.495 \quad Std[\hat q(n)] = 0.091$



$E[\hat u(n)] = 0.499 \quad Std[\hat u(n)] = 0.023$



$E[T(n)] = 1005 \quad Std[T(n)] = 30.9$


## Box/Swarmplots of Performance measures 

   
![image alt text]({static}../images/mm1_queue_simulation_1.png)
    


## Histogram of Performance measures
    
![image alt text]({static}../images/mm1_queue_simulation_2.png)