Title: Singular Value Decomposition
Date: 2020.12.14 
Summary: Theoretical explanation of the SVD. From-scratch Python implementation. Application example on an image.


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
```


```python
plt.style.use("bmh")
config = {'figure.figsize': (16, 4), 
          'axes.titlesize': 18, 
          'axes.labelsize': 14, 
          'lines.linewidth': 2, 
          'lines.markersize': 10, 
          'xtick.labelsize': 10,
          'ytick.labelsize': 10, 
          'axes.prop_cycle': plt.cycler(color=["mediumpurple", "saddlebrown", "darkcyan", "olivedrab", "darkseagreen", "darkkhaki", "darkgoldenrod", "deepskyblue", "firebrick", "palevioletred"]),}
plt.rcParams.update(config)
```

Let $X =$ $\begin{bmatrix}  | & | & | & | \\  x_1 & x_2 & \dotsc & x_m \\  | & | & | & | \end{bmatrix}$ $\in \mathbb{C}^{nxm}$ be a matrix representing data points, where:

* Each row $X_k \in \mathbb{C}^m$ is a variable or dimension of the data.

* Each column $X_k \in \mathbb{C}^n$ is an observation or measurement from simulations or experiments. The index $k$ indicates the $k^{th}$ set of measurements or observations.

The method to be explained is specially useful when $n \gg m$ (more dimensions than observations), i.e. high dimensional problems, resulting in a tall-skinny matrix, as opposed to a short-fat matrix when $n \ll m$ (more observations than dimensions).

The **SVD** is a unique matrix decomposition that exists for every complex-values matrix $X \in \mathbb{C}^{nxm}$:

$$X = U \Sigma V^* \quad (1)$$

where $U \in \mathbb{C}^{n \times n}$ and $V \in \mathbb{C}^{m \times m}$ are unitary matrices$_{[1]}$. When $n \geq m$ the matrix $\Sigma$ has at most $m$ non-zero elements on the diagonal, and may be written as $\Sigma =$ $\begin{bmatrix} \hat{\Sigma} \\ 0 \end{bmatrix}$. Therefore, it is possible to exactly represent $X$ using the **economy SVD**:

$$X = U\Sigma V^* = \hat{U} \hat{\Sigma} V^*$$

where $\hat{U}$ are the first $m$ columns of $U$. The columns of $U$ are called **left singular vectors of X** and the columns of V are **right singular vectors**. The diagonal elements of $\Sigma$, $\sigma_i$, are called **singular values**, and they are ordered so that $\sigma_{i+1} \leq \sigma_i$, i.e., from largest to smallest. The rank of $X$ is equal to the number of non-zero singular values.

<br>

SVD generalizes the eigendecomposition of a square normal matrix to any $m \times n$ matrix via an extension of the polar decomposition. It is numerically stable and provides a hierarchical representation of the data in terms of a new coordinate system defined by dominant correlations within the data. Moreover, the SVD is guaranteed to exist for any matrix, unlike the eigendecomposition.

<br>

**Geometric Interpretation**

If $M$ is an $n \times n$ real square matrix, the matrices $U$ and $V^T$ can be chosen to be real $n \times n$, too, which represent rotations or reflection of the space $R^n$, while $\Sigma$ represents the scaling of each coordinate $X_i$ by the factor $\sigma_i$. Thus the SVD decomposition breaks down any invertible linear transformation of $R^n$ into a composition of three geometrical transformations: a rotation or reflection ($V^*$), a coordinate-by-coordinate scaling ($\Sigma$), and a second rotation or reflection ($U$).

# Interpretation as dominant correlations

The SVD is related to an eigenvalue problem involving the correlation matrices $XX^T$ and $X^TX$:

$$X^T = V \Sigma U^T$$

$$XX^T = U \Sigma V^T V \Sigma U^T = U \hat{\Sigma^2} U^T$$

$$X^TX = V \Sigma U^T U \Sigma V^T = V \hat{\Sigma^2} V^T$$

Recalling that $U$ and $V$ are unitary, multiplying each equation's side by $U$ and $V$, respectively, $U$; $\Sigma$; and $V$ are solutions to the following eigenvalue problems:

$X X^T U = U \hat{\Sigma^2} \quad (2)$

$X^T X V = V \hat{\Sigma^2} \quad (3)$

* The non-negative singular values of $X$ are the square root of the non-negative eigenvalue\s of $X^TX$ and of $XX^T$, and are arranged in descending order by magnitude, and thus the columns of $U$ are hierarchically ordered by how much correlation they capture in the columns of $X$; and the columns of $V$ similarly captures correlation in the rows of $X$.

* The left-singular vectors of $X$ are a set of orthonormal eigenvectors of of the correlation matrix $XX^T$, 

* The right-singular vectors of $X$ are a set of orthonormal eigenvectors of $X^TX$. 

### Notes

**Conjugate transpose**

The conjugate transpose or Hermitian transpose of an $n \times m$ matrix $A$ with complex entries is the $n \times m$ matrix obtained from $A$ by taking the transpose and then taking the complex conjugate of each entry (the complex conjugate of $a+ib$ is $a-ib$, for real numbers $a$ and $b$). It is denoted as $A^H$ or $A^*$. For real matrices, the conjugate transpose is just the transpose, $A^*=A^T$.

<br>

**[1] Unitary vs orthogonal matrix**

In linear algebra, a complex square matrix $U$ is unitary if its conjugate transpose $U^*$ is also its inverse, that is, if $UU^* = U^*U = I$, where $I$ is the identity matrix. For real numbers, unitary and orthogonal means the same.

# Truncated SVD

The sum of the squares of the singular values should be equal to the total variance in $X$, and each one tells how much of the total variance is accounted for by each singular vector. Often, a **truncated SVD** is calculated, truncating the SVD at a rank $r$ that captures a pre-determined amount of the variance or energy in the original matrix:

$$a_{ij} = \sum_{k=1}^{r} u_{ik} \sigma_k v_{jk}$$

## Election of $r$

Deciding how many singular values to keep (i.e. the rank $r$) is one of the most important and contentious decisions when using the SVD. There are many factors, including specifications on the desired rank of the system, the magnitude of noise, and the distribution of the singular values. 

### Percentage of explained variance captured
The SVD can be truncated at a rank $r$ that captures a pre-determined amount of the variance or energy in the original data, such as $90%$ or $99%$ truncation.

### Elbow method
Other techniques involve identifying "elbows" in the singular value distribution, which may denote the transition from singular values that represent important variance percentage from those that represent low information.

### Gavish-Donoho
According to [Gavish-Donoho (2014)](https://arxiv.org/abs/1305.5870), for non squared matrices, the optimal value of $r$ is calculated as $r = \omega(\beta) \cdot median(S)$, where $\beta = \dfrac{n}{m} \; \text{if} \; n < m$ or $\dfrac{m}{n} \; \text{if} \; m < n$, and $\omega(\beta)$ can be found in Table IV in [the original paper](https://arxiv.org/abs/1305.5870).


# Image Approximation


```python
X = imread('piano.jpg') # https://i.pinimg.com/originals/5a/6b/a1/5a6ba131df34eb9ee9a195f18498c839.jpg

# Apply SVD
U, S, VT = np.linalg.svd(X, full_matrices=False)

# Optimal r
beta = min(X.shape) / max(X.shape)
omega = 2.8582
r_optimal = int(np.median(S) * omega)

fig, axes = plt.subplots(3, 3, figsize=(16, 9))
rs = [1, 2,  4, 5, 20, 100, 150, r_optimal, S.size]

for ax, r in zip(axes.reshape(-1), rs):

    # Approximate matrix by truncating in r
    approximation = U[:, :r] @ np.diag(S)[:r, :r] @ VT[:r, :]
    
    # Calculate megabytes used in RAM
    megabytes = (U[:, :r].nbytes + np.diag(S)[:r, :r].nbytes + VT[:r, :].nbytes) / 1024 / 1024
    
    # Calculate variance explained
    variance_explained = np.sum(S[:r]) / np.sum(S) * 100
    
    img = ax.imshow(approximation, aspect='auto')
    ax.set_title(f'{r} Singular Values - {megabytes:.2f} MB - {variance_explained:.2f}% explained', fontsize=13)
    img.set_cmap('gray')
    ax.axis('off')

plt.suptitle('SVD on Image Compression', y=1.08, fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=0.99)
plt.show()
```


    
![image alt text]({static}../images/singular_value_decomposition_1.png)
    



```python
# VARIANCE EXPLAINED
plt.plot(np.cumsum(S) / np.sum(S), color='cadetblue', linewidth=3, zorder=1)
plt.scatter(r_optimal, np.sum(S[:r_optimal]) / np.sum(S), color='saddlebrown', s=200, label='Gavish-Donoho Optimal $r$ Value', zorder=2)

plt.hlines(y=np.sum(S[:r_optimal]) / np.sum(S), xmin=0, xmax=r_optimal, linestyle='--', linewidth=1.5, color='saddlebrown')
plt.vlines(x=r_optimal, ymin=0, ymax=np.sum(S[:r_optimal]) / np.sum(S), linestyle='--', linewidth=1.5, color='saddlebrown')

yticks = np.round(list(np.arange(0,1.1,0.1)) + [np.sum(S[:r_optimal]) / np.sum(S)], 2)
plt.yticks(yticks)

x = np.arange(0, S.size, 100) + 1
xticks = list(np.append(np.array([1]), x[1:] - 1)) + [r_optimal]
plt.xticks(xticks)

plt.ylim(0, 1.05)
plt.xlim(-5, S.size)

plt.xlabel('$r$')
plt.ylabel('Explained Variance')
plt.title('Cumulative Explained Variance', y=1.01)

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```


    
![image alt text]({static}../images/singular_value_decomposition_2.png)
    

# References

[Singular Value Decomposition - Wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition){: target="_blank"}

[Singular Value Decomposition (SVD) Tutorial: Applications, Examples, Exercises](https://blog.statsbot.co/singular-value-decomposition-tutorial-52c695315254){: target="_blank"}

Data Driven Science & Engineering Machine Learning, Dynamical Systems, and Control - Brunton & Kutz

[Singular Value Decomposition - Steve Brunton](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv){: target="_blank"}

[Lecture 47 — Singular Value Decomposition | Stanford University](https://www.youtube.com/watch?v=P5mlg91as1c&t=819s){: target="_blank"}

[Lecture 48 — Dimensionality Reduction with SVD | Stanford University](https://www.youtube.com/watch?v=UyAfmAZU_WI&t=314s){: target="_blank"}