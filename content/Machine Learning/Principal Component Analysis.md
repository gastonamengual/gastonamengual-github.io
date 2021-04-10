Title: Principal Component Analysis
Date: 2021.01.15
Summary: Theoretical explanation of PCA, including Geometric, Covariance, and SVD approaches. From-scratch Python implementation. Application to real dataset. Why PCA should not be used for clustering.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
```


# Definition

**Principal Component Analysis (PCA)** is a dimensionality reduction method that transforms the data into a new orthogonal (i.e. perpendicular, linearly independent, and non-correlated) coordinate system that **keeps most of the variance** that was present in the original data. 

<br>

PCA transforms the data **maximizing the variance** over the projected points. Each new coordinate is called **principal component**, and is ranked according to how much variance they explain, or by **explained variance** (the greatest variance lies on the first principal component, the second greatest variance on the second principal component, ...).

<br>

The data is arranged in a an $n \times m$ matrix $X$, where each row represents a different observation or measurement of the experiment, and each column represents a dimension or feature. 

<br>

The principal components are new features constructed as linear combinations of the initial features, allowing to reduce dimensionality while keeping underlying information structure. Since the principal components are ranked, it is possible to discard those with low contribution (i.e. the added explained variance is negligible). However, these new features are harder to interpret than the original ones, as they are a linear combinations that does not necessarily have an intrinsic meaning.

## Assumptions

In order to apply PCA, the following assumptions must be met.

1. The observations have mean $0$ (i.e., $X$ has column-wise zero empirical mean).

2. The data must be able to be analyzed with a covariance matrix, which in term makes sense when the relation between the data is linear.

3. The sample taken is large enough and there are no significant outliers (heavy-tailed distributions should be avoided).

# Approaches

## Geometric approach

A zero-intercept line is to be found such that the variance of the projected data points over the line is maximized. The principal components represent the directions of the data (projection) that explains the maximal amount of variance, and can be thought as new axes providing the best angle to capture the data dispersion.

Considering a data point, and a line drawn through the origin, then:

* Let $a$ be the position vector of the point. 
* Let $b$ be the perpendicular distance vector from the point to the line.
* Let $c$ be the position vector of the projected point.

<img src="https://liorpachter.files.wordpress.com/2014/05/pca_figure1.jpg" style="width: 350px;" />

All three vectors form a square triangle such that, according to the Pythagoras theorem, $a^2 = b^2 + c^2$. $b$ and $c$ are inversely related and, since $a$ is fixed, if $b$ gets bigger, $c$ gets smaller, and vice versa. As the objective is to maximize variance (dispersion), PCA can either minimize the distance to the line ($b$) or maximize the distance from the projected point to the origin ($c$).

For more information, relate to [Principal Component Analysis (PCA), Step-by-Step - StatQuest](https://www.youtube.com/watch?v=FgakZw6K1QQ&feature=emb_title)

## Covariance approach

As an alternative, the principal components can be also thought as eigenvectors of the data covariance matrix, and can be computed by eigendecomposition of the data covariance matrix. 

Let $X$ be an $n \times m$ matrix where each row represents a different observation or measurement of the experiment, and each column a dimension or feature. 

1. **Calculate $B$, an $n \times m$ matrix where each column is computed by subtracting the column-wise mean $\bar{X}$ from each column of $X$, namely, $B = X - \bar{X}$.**

2. **Calculate the covariance matrix $C$**, such that $C = \dfrac{B^TB}{n-1}$.

3. **Compute the eigenvectors ($V$) and eigenvalues ($\Lambda$) of $C$**, such that $CV = \Lambda V$.

4. **Sort the eigenvectors according to their eigenvalues in decreasing order**.

5. **Choose the first $r$ eigenvectors that will be the new $r$ dimensions**. Select a subset $W$ of the eigenvectors as basis vectors.

6. **Compute the transformed data matrix $T$**, such that $T = B W$.


```python
def principal_component_analysis_covariance(dataset):
    
    # The function takes a Pandas DataFrame
    X = dataset.copy()
    
    # 1 Calculate B: Center data
    B = X - X.mean()
    
    B = B.to_numpy()
    
    # 2 Calculate C: Covariance matrix
    C = (1 / (B.shape[0] - 1)) * (B.T @ B)

    # 3 Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(C)
    eigenvectors = eigenvectors.T

    # 4 Sort eigenvectors and eigenvalues
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]
    
    # The 5th step will be performed outside this function. The full transformed data matrix will be returned, and the r principal 
    # components will be chosen later. If desired, this function could take the number of components as a parameter, and the subset
    # of the eigenvectors can be selected here. 
    
    # 6 Compute transformed data matrix and atributes
    components = eigenvectors
    explained_variance = eigenvalues
    explained_variance_ratio = explained_variance / explained_variance.sum()
    singular_values = np.sqrt(explained_variance * (B.shape[0] - 1)) # Explained in section 4
    transformed_data = B @ components.T
    
    return {'components': components, 
            'explained_variance': explained_variance, 
            'explained_variance_ratio': explained_variance_ratio, 
            'singular_values': singular_values, 
            'transformed_data': transformed_data,}
```

## SVD approach

Another approach based on eigendecomposition is calculating PCA as a subset of the Singular Value Decomposition (SVD). Let $X$ be an $n \times m$ matrix where each row represents a different observation or measurement of the experiment, and each column a dimension or feature. 

1. **Calculate $B$, an $n \times m$ matrix where each column is computed by subtracting the column-wise mean $\bar{X}$ from each column of $X$, namely, $B = X - \bar{X}$.**

2. **Compute the SVD**. $B = U \Sigma V^T$ 

3. **Choose the first $r$ rows from $U$ and first $r$ singular (new $r$ dimensions)**.

4. **Compute the transformed data matrix $T$**, such that $T = XV = U \Sigma V^T V = U \Sigma$. This form is also the polar decomposition of $T$.

<br>

(For more information on SVD, relate to [Singular Value Decomposition](https://gastonamengual.github.io/singular-value-decomposition.html){: target="_blank"})


```python
def principal_component_analysis_svd(dataset):
    
    # The function takes a Pandas DataFrame
    X = dataset.copy()
    
    # 1 Calculate B: Center data
    B = X - X.mean()
    
    B = B.to_numpy()
    
    # 2 Compute SVD
    U, S, VT = np.linalg.svd(B, full_matrices=False)

    # The 3rd step will be performed outside this function. The full transformed data matrix will be returned, and the r principal 
    # components will be chosen later. If desired, this function could take the number of components as a parameter, and the subset
    # of the eigenvectors can be selected here. 
    
    # 4 Compute transformed data matrix and atributes
    components = VT
    explained_variance = S**2 / (B.shape[0] - 1) # Explained in section 4
    explained_variance_ratio = explained_variance / explained_variance.sum()
    singular_values = S
    transformed_data = U @ np.diag(S)
    
    return {'components': components, 
            'explained_variance': explained_variance, 
            'explained_variance_ratio': explained_variance_ratio, 
            'singular_values': singular_values, 
            'transformed_data': transformed_data,}
```

### Relation between Covariance and SVD approaches 

According to the SVD, 

$$B = U \Sigma V^T$$

$$B^TB = V \Sigma U^T U \Sigma V^T = V \Sigma^2 V^T$$

The covariance matrix of $B$ is $C = \dfrac{B^TB}{n-1} = \dfrac{V \Sigma^2 V^T}{n-1}$.

$\Sigma$ are the eigenvalues of the matrix $B$, while $\Lambda$ are the eigenvalues of $C$. The relationship between them is defined as $\Lambda = \dfrac{\Sigma^2}{n-1}$.

Moreover, the right singular vectors $V$ of $B$ are equivalent to the eigenvectors of $B^TB$.

# Application

PCA will be applied to the ozone data set (Forecasting skewed biased stochastic ozone days: analyses, solutions and beyond, Knowledge and Information Systems, Vol. 14, No. 3, 2008). This data set contains 2534 observations and 73 features. 

**Author**: Kun Zhang, Wei Fan, XiaoJing Yuan. **Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/ozone+level+detection).


### Import data set


```python
dataset = pd.read_csv('datasets/principal_component_analysis_ozone.csv')
dataset.head()
```

<div>
<div class="table-div">
<table border="1">
  <thead>
    <tr>
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V64</th>
      <th>V65</th>
      <th>V66</th>
      <th>V67</th>
      <th>V68</th>
      <th>V69</th>
      <th>V70</th>
      <th>V71</th>
      <th>V72</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8</td>
      <td>1.8</td>
      <td>2.4</td>
      <td>2.1</td>
      <td>2.0</td>
      <td>2.1</td>
      <td>1.5</td>
      <td>1.7</td>
      <td>1.9</td>
      <td>2.3</td>
      <td>...</td>
      <td>0.15</td>
      <td>10.67</td>
      <td>-1.56</td>
      <td>5795</td>
      <td>-12.1</td>
      <td>17.9</td>
      <td>1033</td>
      <td>-55</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.8</td>
      <td>3.2</td>
      <td>3.3</td>
      <td>2.7</td>
      <td>3.3</td>
      <td>3.2</td>
      <td>2.9</td>
      <td>2.8</td>
      <td>3.1</td>
      <td>3.4</td>
      <td>...</td>
      <td>0.48</td>
      <td>8.39</td>
      <td>3.84</td>
      <td>5805</td>
      <td>14.05</td>
      <td>29</td>
      <td>10275</td>
      <td>-55</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.9</td>
      <td>2.8</td>
      <td>2.6</td>
      <td>2.1</td>
      <td>2.2</td>
      <td>2.5</td>
      <td>2.5</td>
      <td>2.7</td>
      <td>2.2</td>
      <td>2.5</td>
      <td>...</td>
      <td>0.6</td>
      <td>6.94</td>
      <td>9.8</td>
      <td>5790</td>
      <td>17.9</td>
      <td>41.3</td>
      <td>10235</td>
      <td>-40</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.7</td>
      <td>3.8</td>
      <td>3.7</td>
      <td>3.8</td>
      <td>2.9</td>
      <td>3.1</td>
      <td>2.8</td>
      <td>2.5</td>
      <td>2.4</td>
      <td>3.1</td>
      <td>...</td>
      <td>0.49</td>
      <td>8.73</td>
      <td>10.54</td>
      <td>5775</td>
      <td>31.15</td>
      <td>51.7</td>
      <td>10195</td>
      <td>-40</td>
      <td>2.08</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.6</td>
      <td>2.1</td>
      <td>1.6</td>
      <td>1.4</td>
      <td>0.9</td>
      <td>1.5</td>
      <td>1.2</td>
      <td>1.4</td>
      <td>1.3</td>
      <td>1.4</td>
      <td>...</td>
      <td>0.30</td>
      <td>9.87</td>
      <td>0.83</td>
      <td>5818</td>
      <td>10.51</td>
      <td>37.38</td>
      <td>10164</td>
      <td>-0.119</td>
      <td>0.58</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

<br>
<p>2534 rows × 73 columns</p>



### Correlation Matrix


![image alt text]({static}../images/principal_component_analysis_1.png)
    


### Perform Dimensionality Reduction


```python
pca = principal_component_analysis_covariance(dataset)
```


```python
x_ax = np.arange(1, pca['explained_variance_ratio'].size + 1, 1)
plt.plot(x_ax, pca['explained_variance_ratio'])
```


    
![image alt text]({static}../images/principal_component_analysis_2.png)
    


The plot above is a **scree plot**, and it shows the contribution of each principal component to the explained variance (ratio). It displays the principal components in descending order of eigenvalues or explained variance. It is useful to analyze which principal components capture most of the variance of the original data. In this case, it can be observed that beyond 5 principal components, no significant amount of data is explained.


```python
# Step 5 performed after transforming data.
n_components = 5

explained_variance_n_components = explained_variance_ratio[:n_components]
transformed_data_n_components = pca['transformed_data'][:,:n_components]

print(f'The first {n_components} components explain {explained_variance_n_components.sum():.3g}% of the original dataset variance')
print(f'{n_components / dataset.shape[1]:.3g}% of the original number of columns are kept')

columns = [f'PC{i + 1}' for i in range(n_components)]

pd.DataFrame(transformed_data_n_components, columns=columns).head()
```

    The first 5 components explain 0.976% of the original dataset variance
    0.0685% of the original number of columns are kept
    




<div class="table-div">
<table border="1">
  <thead>
    <tr>
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.626476</td>
      <td>-174.944631</td>
      <td>103.961564</td>
      <td>2.215012</td>
      <td>-9.158727</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.753434</td>
      <td>-107.886011</td>
      <td>89.561838</td>
      <td>22.584096</td>
      <td>-6.403647</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-7.291178</td>
      <td>-66.648771</td>
      <td>63.710631</td>
      <td>27.268419</td>
      <td>-4.361848</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-31.119019</td>
      <td>-22.354213</td>
      <td>53.012715</td>
      <td>37.152561</td>
      <td>4.960774</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.369718</td>
      <td>0.427355</td>
      <td>-0.015712</td>
      <td>0.438565</td>
      <td>-0.936091</td>
    </tr>
  </tbody>
</table>
</div>



### Resulting Correlation Matrix
    
![image alt text]({static}../images/principal_component_analysis_3.png)
    


The previous Correlogram verifies how PCA has successfully transformed a correlated data matrix into a linearly independent data matrix, i.e. uncorrelated. The question of how many principal components to keep will highly depend on deciding what matters most, the size the matrix takes or the amount of variance explained, according to the requirements of the application.

## Covariance and SVD differences


```python
covariance_components = principal_component_analysis_covariance(dataset)['components']
svd_components = principal_component_analysis_svd(dataset)['components']
```


```python
assert np.mean(np.isclose(np.abs(covariance_components), np.abs(svd_components))) == 1

print('The absolute values of the components of both approaches match')
```

The absolute values of the components of both approaches match
    


```python
percentage_positive = np.mean(np.isclose(covariance_components, svd_components))

print(f'{percentage_positive:.3g}% of the values of SVD are equally signed with Covariance values')

percentage_positive = np.mean(np.isclose(covariance_components, -svd_components))

print(f'{percentage_positive:.3g}% of the values of SVD have opposite sign than Covariance values')
```

0.63% of the values of SVD are equally signed with Covariance values
0.37% of the values of SVD have opposite sign than Covariance values
    


```python
assert np.mean(np.isclose(principal_component_analysis_covariance(dataset)['explained_variance'], principal_component_analysis_svd(dataset)['explained_variance'])) == 1

assert np.mean(np.isclose(principal_component_analysis_covariance(dataset)['singular_values'], principal_component_analysis_svd(dataset)['singular_values'])) == 1

print(f'Both approaches have equal explained variance and singular values')
```

Both approaches have equal explained variance and singular values

<br> 

As it can be observed, the principal components of both Covariance and SVD and covariance approaches differ in sign 37% of the times. No further analysis nor examination on the cause of this situation was performed.

# PCA and Classification/Clustering

Principal Component Analysis is an unsupervised learning algorithm that, as discussed in the first section, transforms the data maximizing the variance over the projected points. In such process, there is no attempt to separate the data based on their labels, and therefore PCA will fail to divide the data according to which class each point belongs to. 

The Principal-Component-Analysis-like supervised learning algorithm for classification is called Linear Discriminant Analysis. For more information, relate to [Discriminant Analysis](https://gastonamengual.github.io/discriminant-analysis.html){: target="_blank"}.

# References

[A step-by-step Explanation of Principal Component Analysis](https://builtin.com/data-science/step-step-explanation-principal-component-analysis){: target="_blank"}

[PCA From Scratch in Python](https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51){: target="_blank"}

[How to Calculate Principal Component Analysis (PCA) from Scratch in Python](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/){: target="_blank"}

[Sklearn PCA Implementation](https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/decomposition/pca.py#L385){: target="_blank"}

[Assumptions](https://statistics.laerd.com/spss-tutorials/principal-components-analysis-pca-using-spss-statistics.php){: target="_blank"}

[PCA (Principal Component Analysis) in Python -  Python Engineer](https://www.youtube.com/watch?v=52d7ha-GdV8&t=3s){: target="_blank"}

[Principal Component Analysis (PCA), Step-by-Step - StatQuest](https://www.youtube.com/watch?v=FgakZw6K1QQ&feature=emb_title){: target="_blank"}
