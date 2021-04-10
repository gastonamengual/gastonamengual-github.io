Title: Discriminant Analysis
Date: 2021.01.27
Summary: Linear and Quadratic Discriminant Analysis. Applications of LDA (classification, dimensionality reduction) and differences with PCA. Application of QDA.

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.decomposition import PCA
```

# Definition

Linear Discrimanant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) are two algorithms used for classification with a linear and a quadratic decision surface, respectively. Both classifiers have closed-form solutions that can be easily computed, are inherently multiclass, and have no hyperparameters to tune.

Both LDA and QDA are derived from simple probabilistic models wich model the class conditional distribution of the data $P(X | y=k)$ for each class $k$. For each observation $x$, predictions can be obtained using Bayes' theorem

$$P(y=k | x) = \dfrac{P(x | y=k) P(y=k)}{P(x)} = \dfrac{P(x | y=k) P(y = k)}{ \sum_{l} P(x | y=l) \cdot P(y=l)}$$

The class $k$ which maximizes the posterior probability is selected as a prediction. The model makes the assumption that $P(x|y)$ is modeled as a multivariate Gaussian distribution with density

$$P(x | y=k) = \frac{1}{(2\pi)^{m/2} |\Sigma_k|^{1/2}}\exp\left(-\frac{1}{2} (x-\mu_k)^t \Sigma_k^{-1} (x-\mu_k)\right)$$

where $m$ is the number of features.

## Quadratic Discriminant Analysis

Then, the log of the posterior is 

$$\begin{split}\log P(y=k | x) &= \log P(x | y=k) + \log P(y = k) + Cst \\
&= -\frac{1}{2} \log |\Sigma_k| -\frac{1}{2} (x-\mu_k)^t \Sigma_k^{-1} (x-\mu_k) + \log P(y = k) + Cst,\end{split}$$

where the constant term $Cst$ corresponds to the denominator $P(x)$, in addition to other constant terms from the Gaussian. The predicted class is the one that maximises this log-posterior. The quadratic term lies in the multiplication of $x \cdot \Sigma_k^{-1} \cdot x = x^2 \cdot \Sigma_k^{-1}$.

If it is assumed that the covariance matrices are diagonal, the inputs are assumed to be conditionally independent in each class, and the classifier is equivalente to the Gaussian Naive Bayes.

## Linear Discriminant Analysis

LDA is a special case of QDA where the Gaussian distributions for each class are assumed to have the same covariance matrix, that is $\Sigma_k - \Sigma$ for all $k$. This reduces the log posterior to 

$$\log P(y=k | x) = -\frac{1}{2} (x-\mu_k)^t \Sigma^{-1} (x-\mu_k) + \log P(y = k) + Cst$$

LDA has a linear decision surface, while QDA has quadratic decision surfaces.

### LDA for Dimensionality Reduction

LDA can also be used to perform supervised dimensionality reduction, by projecting the input data to a linear subspace consisting of the directions which maximize the separation between classes. The dimension of the output is necessarily less than the number of classes, so this is in general a rather strong dimensionality reduction, and only makes sense in a multiclass setting.

# LDA for Classification


```python
# Data set
wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
```


```python
# Train
lda = LDA(n_components=2).fit(X_train, y_train)
# Test
predictions = lda.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
```

    Accuracy: 1.0
    

**Prediction Map on Two Data Sets**


```python
datasets = [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1)]

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

for ax, dataset in zip(axes, datasets):
    X_, y_ = dataset
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=.4, random_state=42)

    x_min, x_max = X_[:, 0].min() - .5, X_[:, 0].max() + .5
    y_min, y_max = X_[:, 1].min() - .5, X_[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .001), np.arange(y_min, y_max, .001))

    lda = LDA().fit(X_train, y_train)
    score = lda.score(X_test, y_test)

    Z = lda.decision_function(np.column_stack((xx.flatten(), yy.flatten())))
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy)
    ax.scatter(*X_.T)

```

![image alt text]({static}../images/discriminant_analysis_1.png)
    


# LDA for Dimensionality Reduction

The main difference between LDA and PCA is that the first focus on maximizing the separation between multiple classes, while the latter focus on finding the components that maximize the variance of the data. Moreover, LDA is a supervised method what uses known class labels, unlike PCA, which is an unsupervised learning algorithm

![image alt text]({static}../images/discriminant_analysis_2.png)


```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

X_transformed_LDA = LDA(n_components=2).fit_transform(X, y)
axes[0].scatter(*X_transformed_LDA.T)

X_transformed_PCA = PCA(n_components=2).fit_transform(X)                
axes[1].scatter(*X_transformed_PCA.T)
```

    
![image alt text]({static}../images/discriminant_analysis_3.png)
    


It can be seen that LDA successfuly reduces the original dimension (13) into two dimension, while at the same time separates the point of different class. PCA, however, does not intend to separate point in classes, but to maximize the variance explained by each principal component. Furthermore, it can be seen that all three classes can be separated by a straight vertical line.

# LDA as input for other classification algorithm

When a data set has multiple features, it is not possible to visualize the results, and one only counts with the calculation of different metrics to evaluate the performance of the algorithm. However, LDA can be applied as an input of another algorithm to reduce the dimensionality of the data while maintaining separability between classes, in order to generate, for example, a visual prediction map, which would not be possible in more than 2 or 3 dimensions, and adds value to the report of the predictions.

The *k*-NN algorithm will be applied to the transformed data from the previous section.


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
x_min, x_max = X_transformed_LDA[:, 0].min() - .5, X_transformed_LDA[:, 0].max() + .5
y_min, y_max = X_transformed_LDA[:, 1].min() - .5, X_transformed_LDA[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_transformed_LDA, y)

Z = knn.predict(np.column_stack((xx.flatten(), yy.flatten())))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z)
plt.scatter(*X_transformed_LDA.T)
```


    
![image alt text]({static}../images/discriminant_analysis_4.png)
    


# QDA for Classification


```python
datasets = [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1)]

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

for ax, dataset in zip(axes, datasets):
    
    X_, y_ = dataset
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=.4, random_state=42)

    x_min, x_max = X_[:, 0].min() - .5, X_[:, 0].max() + .5
    y_min, y_max = X_[:, 1].min() - .5, X_[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .001), np.arange(y_min, y_max, .001))

    qda = QDA().fit(X_train, y_train)
    score = qda.score(X_test, y_test)

    Z = qda.decision_function(np.column_stack((xx.flatten(), yy.flatten())))
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z)
    ax.scatter(*X_.T)
```


    
![image alt text]({static}../images/discriminant_analysis_5.png)
    


# References

[Sklearn - Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html){: target="_blank"}

[StatQuest: Linear Discriminant Analysis](https://www.youtube.com/watch?v=azXCzI57Yfc){: target="_blank"}

[Comparison of LDA and PCA 2D projection of Iris dataset](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html){: target="_blank"}
