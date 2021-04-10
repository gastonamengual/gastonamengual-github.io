Title: Logistic Regression
Date: 2021.01.19
Summary: Definition of Logistic Regression. Demonstration and explanation of cost function. Optimization with Gradient Descent. From-scratch implementation in Python. Application on dummy data.

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
```


# Definition

Logistic regression is a statistical model that uses a logistic function to model a binary dependent variable in order to estimate the parameters of a logistic model. The goal of binary logistic regression is to train a classifier that can make a binary decision about the class of a new input observation. 

# Model

Let $X$ be an $n \times m$ matrix with $n$ observations and $m$ independent variables or features, $x$ an individual observation $1 \times m$ vector, $y$ the true known output for each observation (which is $0$ or $1$), and $\hat{y}$ be the prediction of the model for an observation. 

Logistic regression makes a prediction by learning a vector of weights, $w$, and a bias term, $b$. Each weight $w_i$ is a real number, and is associated with one of the observation's features $x_{nm}$. To make a decision, after the weights and bias have been learned, the the weighted sum of the evidence for the class, $z$ is calculated:

$$z = \left ( \sum_{i=1}^{n} w_i x_i \right ) + b = w \cdot x + b$$

The domain of $z$ is $\left [ -\infty, \infty \right ]$. However, the output of the model is interpreted as the probability of the observation belonging to class 1. As the probability is a real number between 0 and 1, the output of the model must be squashed between 0 and 1. To map from $\left [ -\infty, \infty \right ]$ to $\left [ 0, 1 \right ]$, a sigmoid or logistic function $\sigma(z)$ is used. The sigmoid has the following equation:

$$\hat{y} = \sigma(z) = \dfrac{1}{1+e^{-z}}$$

Why is the sigmoid used?

* It takes real values
* It maps the values into the range $[0, 1]$
* It tends to squash outlier values toward 0 or 1
* It is differentiable


```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```


```python
z = np.linspace(-15, 15)
plt.plot(z, sigmoid(z), lw=3)
plt.axhline(0, ls='--', lw=1, c='black')
plt.axhline(1, ls='--', lw=1, c='black')
plt.xlim(-15, 15)
plt.title('Sigmoid Function')
plt.tight_layout()
plt.show()
```


    
![image alt text]({static}../images/logistic_regression_1.png)
    


Applying the sigmoid function to $z$ produces a number between 0 and 1. To make it a probability, it will be verified that the two cases, $p(y = 1)$ and $p(y = 0)$ sum to 1:

$$p(y = 1) = \sigma(w \cdot x + b) \qquad \qquad p(y = 0) = 1 - \sigma(w \cdot x + b)$$

$$p(y = 1) + p(y = 0) = \sigma(w \cdot x + b) + 1 - \sigma(w \cdot x + b) = 1$$

Finally, a decision is made as follows

$$\hat{y} = \left\{\begin{matrix} 1 & \text{if} \; P(y=1 | x) > 0.5\\ 0 & \text{otherwise} \end{matrix}\right.$$

where $0.5$ is called the decision boundary.

# Learning Parameters

### Cross-entropy Loss Function

To express how close the classifier output $\hat{y}$ is to the true output $y$, a loss function $L(\hat{y}, y)$ that prefers the correct class labels of the training examples to be more likely. This **conditional maximum likelihood estimation** chooses the parameters that maximize the log probability of the true $y$ labels in the training data given the observations $x$. The resulting loss function is the negative log likelihood loss, generally called the **cross-entropy loss**. This loss function is a Bernoulli distribution, as there are only to discrete outcomes:

$$p(y | x) = \hat{y}^y (1 - \hat{y})^{1 - y}$$

Taking the log on both sides is mathematically handy, and whatever values maximize a probability will also maximize the log of the probability.

$$log p(y | x) = log \left [ \hat{y}^y (1 - \hat{y})^{1 - y} \right ] = y \; log \hat{y} + (1 - y) \; log (1 - \hat{y})$$

In order to minimize the log likelihood, the sign is fliped:

$$L_{CE} = - log p(y | x) = - \left [ y \; log \hat{y} + (1 - y) \; log (1 - \hat{y}) \right ]$$

Finally, plugging the definition of $\hat{y}$, the loss function is defined as:

$$L_{CE} = - \left [ y \; log \; \sigma(w \cdot x + b) + (1 - y) \; log (1 - \sigma(w \cdot x + b)) \right ]$$

This loss function is convex, and therefore has just one minimum.

### Gradient Descent

In order to find the optimum parameters, gradient descent will be used, as, because the loss function is convex, it is guaranteed to find the minimum. Let $\theta = w, b$ and $f(x_i ; \theta) = \hat{y}$. The goal is to find the set of weights which minimizes the loss function, averaged over all examples:

$$\hat{\theta} = \underset{\theta}{\text{argmin}} \dfrac{1}{m} \sum_{i=1}^{m} L_{CE} (f(x_i ; \theta), y_i)$$

In order to update $\theta$, the gradient $\nabla L(f(x_i ; \theta)$ must be defined. The partial derivatives of the cross-entropy loss function are

$$\frac{\partial L_{CE} (\hat{y}, y)}{\partial w_j} = \left [ \sigma(w \cdot x + b) - y \right ] x_j$$

$$\frac{\partial L_{CE} (\hat{y}, y)}{\partial b} = \sigma(w \cdot x + b) - y$$

# Application

### Create Dummy Data Set


```python
n_samples = 200

np.random.seed(0)

x = np.random.normal(size=n_samples)
y = (x > 0).astype(float)

x[x > 0] *= 3
x += .3 * np.random.normal(size=n_samples)

plt.scatter(x, y)
```


    
![image alt text]({static}../images/logistic_regression_2.png)
    


### Train and Test Split


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

train_data = np.column_stack((x_train, y_train))
test_data = np.column_stack((x_test, y_test))
```

### Training: Gradient Descent Optimization

Stochastic Gradient Descent will be used with a learning rate of $0.0001$ and $7000$ epochs.


```python
costs = []
num_observations = train_data.shape[0]
num_features = 1

learning_rate = 0.0001
num_epochs = 3000
b = 0
w = 0

for _ in tqdm(range(num_epochs)):

    shuffled_data = np.random.permutation(train_data)
    for x_, y_ in shuffled_data:
        
        sigma = sigmoid(x_ * w + b)
        
        dLdw = (sigma - y_) * x_
        dLdb = sigma - y_

        w -= learning_rate * dLdw
        b -= learning_rate * dLdb
        
    cost = - 1 / num_observations * np.sum( y_ * np.log(sigma) + (1 - y_) * np.log(1 - sigma) )
    costs.append(cost)
```
    


```python
plt.plot(costs)
```


    
![image alt text]({static}../images/logistic_regression_3.png)
    



```python
display(Markdown(f'$z = {w:.3f} x {b:.3f}$'))
```


$z = 3.151 x -0.351$



```python
y_hat = sigmoid(w*x_train + b)
y_hat[y_hat > .5] = 1
y_hat[y_hat <= .5] = 0

train_accuracy = np.mean(y_hat == y_train)
print(f'Accuracy for Train Set: {train_accuracy}')
```

    Accuracy for Train Set: 0.95625
    

###  Testing


```python
y_hat = sigmoid(w*x_test + b)
y_hat[y_hat > .5] = 1
y_hat[y_hat <= .5] = 0

test_accuracy = np.mean(y_hat == y_test)
print(f'Accuracy for Test Set: {test_accuracy}')
```

    Accuracy for Test Set: 0.95
    

### Predictions for Whole Data Set


```python
y_hat = sigmoid(w*x + b)
y_hat[y_hat > .5] = 1
y_hat[y_hat <= .5] = 0
```


```python
plt.scatter(x, y)

x_space = np.linspace(x.min(), x.max())
plt.plot(x_space, sigmoid(w*x_space + b))
plt.axhline(0.5)
```


    
![image alt text]({static}../images/logistic_regression_4.png)
    



```python
dataset_accuracy = np.mean(y_hat == y)
print(f'Accuracy for Whole Data Set: {dataset_accuracy}')
```

    Accuracy for Whole Data Set: 0.955
    

# References

[Logistic Regression - Speech and Language Processing. Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/5.pdf){: target="_blank"}
