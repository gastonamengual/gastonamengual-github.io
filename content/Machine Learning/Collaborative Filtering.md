Title: Collaborative Filtering
Date: 2021.01.07
Summary: Definition. Data Preparation. Application of Collaborative Filtering with Stochastic Mini-Batch Gradient Descent, L2 Regularization and Cross-validation, using Python.

**Notebook written by Gastón Amengual and <a href="https://elc.github.io" target="_blank">Ezequiel L. Castaño</a> as the base for Watch This! app (see second portfolio project).**

<hr>

**Collaborative filtering** is a technique used by recommender systems. It is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborative). The underlying assumption is that if a person A has the same opinion as person B on an issue, A is more likely to have B's opinion on a different issue than that of a randomly chosen person.

Collaborative filtering approaches often suffer from three problems:

* Cold start: For a new user or item, there isn't enough data to make accurate recommendations.
* Scalability: In many of the environments in which these systems make recommendations, there are millions of users and products. Thus, a large amount of computation power is often necessary to calculate recommendations.
* Sparsity: The number of items sold on major e-commerce sites is extremely large. The most active users will only have rated a small subset of the overall database. Thus, even the most popular items have very few ratings.

In the **model-based approach approach**, models are developed using different data mining, machine learning algorithms to predict an user's rating of unrated items, and they involve a step to reduce or compress the large but sparse (matrix in which most of the cells are empty) user-item matrix. If the matrix is mostly empty, reducing dimensions can improve the performance of the algorithm in terms of both space and time.

A matrix with dimensions $m \times n$ can be reduced to a product of two matrices $U$ and $V$ with dimensions $m \times p$ and $p \times n$, respectively. The reduced matrices represent the users and items individually. The rows in the first matrix represent the users, and the columns, the features or characteristics of the users. The same applies for the item matrix. These rows or columns are called **latent factors** and are an indication of hidden characteristics about the users or the items. The number of latent factors affects affects the recommendations in a manner where the greater the number of factors, the more personalized the recommendations become, but too many factors can lead to overfitting in the model. This number must be optimized during the training of the model. 

For matrix decomposition an approximation of **Singular Value Decomposition (SVD)** will be used, using stochastic mini-batch gradient descent. The model will be trained using 5-fold cross-validation.


```python
movie_lens = pd.read_csv(r"C:\Users\Gastón\Office\Data Science\_Datasets\movie_lens_100k_data.csv", sep='\t').drop_duplicates().drop("timestamp", axis=1)
```


```python
max_movie_limits = 200
max_user_limits = 50

random_seed = 42

num_epochs = 50
alpha = 0.01
r = None

folds = 5
lambdas = np.linspace(0, 0.3, 21)[1:]
mini_batch_sizes = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
```

# Data Preparation

Both training and testing data must have the same dimensions in order to approximate SVD. 

## Filtering

The most frequent movies and users are kept, as a user who has not seen many movies (or a movie that has not been seen by many users) contributes little to the training or testing, as it acts like a new movie/user.


```python
# Keep most N frequent Movies
movies_by_count = movie_lens.groupby("movieId", as_index=False).size().sort_values(by="size", ascending=False)["movieId"].to_numpy()
movies_available = movies_by_count[:max_movie_limits]
movie_lens = movie_lens[movie_lens['movieId'].isin(movies_available)]

# Keep most M frequent Users
users_by_count = movie_lens.groupby("userId", as_index=False).size().sort_values(by="size", ascending=False)["userId"].to_numpy()
users_available = users_by_count[:max_user_limits]
movie_lens = movie_lens[movie_lens['userId'].isin(users_available)]

movie_lens = movie_lens.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# Verification
uniques = movie_lens.nunique()
assert uniques["userId"] == max_user_limits and uniques["movieId"] == max_movie_limits  # Limits are preserved
```

## Build Minimal Train and Test Data Sets

It is sought that at least one user-movie pair is in the data sets, such that there is at most N unique users and M unique movies. It is also checked if an $n \times m$ test data set can be built. 


```python
# Extracted from: https://stackoverflow.com/a/44115314/7690767
train_unique_user_rows = movie_lens.groupby('userId', group_keys=False).apply(lambda x: x.sample(1, random_state=random_seed))

train_unique_movie_rows = movie_lens.groupby('movieId', group_keys=False).apply(lambda x: x.sample(1, random_state=random_seed))
train_unique_movie_rows = train_unique_movie_rows[~ train_unique_movie_rows["movieId"].isin(train_unique_user_rows["movieId"])]

train_minimal = pd.concat([train_unique_movie_rows, train_unique_user_rows])

# Verification
uniques = train_minimal.nunique()
assert uniques["userId"] == max_user_limits and uniques["movieId"] == max_movie_limits  # Limits are preserved
```


```python
test_base = movie_lens.drop(index=train_minimal.index)
test_unique_user_rows = test_base.groupby('userId', group_keys=False).apply(lambda x: x.sample(1, random_state=random_seed))

test_unique_movie_rows = test_base.groupby('movieId', group_keys=False).apply(lambda x: x.sample(1, random_state=random_seed))
test_unique_movie_rows = test_unique_movie_rows[~ test_unique_movie_rows["movieId"].isin(test_unique_user_rows["movieId"])]

test_minimal = pd.concat([test_unique_movie_rows, test_unique_user_rows])

# Verification
uniques = test_minimal.nunique()
assert uniques["userId"] == max_user_limits and uniques["movieId"] == max_movie_limits  # Limits are preserved
assert not test_minimal.isin(train_minimal).any().any()  # Train and Test have 0 elements in common
```

## Build Train and Validation Data Sets

Create train and validation data sets. The train set consists of the minimal train data set plus a random 80% of the non used rows of the data set with most frequent movies and users, and the validation data set consists of the remaining 20%.


```python
non_used_rows = movie_lens.drop(index=train_minimal.index).drop(index=test_minimal.index)

random_sample = non_used_rows.sample(frac=0.9, random_state=random_seed)

movie_lens_train = pd.concat([train_minimal, random_sample])
movie_lens_validation = movie_lens.drop(index=movie_lens_train.index)

# Verification
assert train_minimal.isin(movie_lens_train).all().all()     # All train minimal entries are in train dataset
assert test_minimal.isin(movie_lens_validation).all().all()          # All test minimal entries are in test dataset
assert not movie_lens_train.isin(movie_lens_validation).any().any()  # Train and Test have 0 elements in common
assert movie_lens_train.shape[0] + movie_lens_validation.shape[0] == movie_lens.shape[0]  # Shapes sum whole dataset
```

## Feature Matrices

Consisting of pivoting the train and validation matrices.


```python
# Pivoting Tables
movie_lens_train_matrix = movie_lens_train.pivot(index='userId', columns='movieId', values='rating')
movie_lens_validation_matrix = movie_lens_validation.pivot(index='userId', columns='movieId', values='rating')

# Verification
assert movie_lens_train_matrix.shape == movie_lens_validation_matrix.shape   # Train and Test have idential shape
assert movie_lens_train_matrix.shape == (max_user_limits, max_movie_limits)  # Limits are preserved

values_in_train = list(zip(*np.where(~np.isnan(movie_lens_train_matrix))))
values_in_validation = list(zip(*np.where(~np.isnan(movie_lens_validation_matrix))))

assert not any(value_train in values_in_validation for value_train in values_in_train)
```

# Cross Validation (K-Folds)


```python
def generate_cross_validated_sample(base_matrix, seed=None, folds=5):    
    repository = base_matrix.copy()
    
    # Initial shuffle
    repository = repository.sample(frac=1, random_state=seed)
    
    # Create minimaml matrices
    train_minimals = []
    test_minimals = []
    
    for _ in range(folds):
        train_unique_user_rows = repository.groupby('userId', group_keys=False).apply(lambda x: x.sample(1, random_state=seed))

        train_unique_movie_rows = repository.groupby('movieId', group_keys=False).apply(lambda x: x.sample(1, random_state=seed))
        train_unique_movie_rows = train_unique_movie_rows[~ train_unique_movie_rows["movieId"].isin(train_unique_user_rows["movieId"])]

        train_minimal = pd.concat([train_unique_movie_rows, train_unique_user_rows])
        train_minimals.append(train_minimal)
        
        repository = repository.drop(index=train_minimal.index)
        
            
    for _ in range(folds):
        test_unique_user_rows = repository.groupby('userId', group_keys=False).apply(lambda x: x.sample(1, random_state=seed))

        test_unique_movie_rows = repository.groupby('movieId', group_keys=False).apply(lambda x: x.sample(1, random_state=seed))
        test_unique_movie_rows = test_unique_movie_rows[~ test_unique_movie_rows["movieId"].isin(test_unique_user_rows["movieId"])]

        test_minimal = pd.concat([test_unique_movie_rows, test_unique_user_rows])
        test_minimals.append(test_minimal)
        
        repository = repository.drop(index=test_minimal.index)
        
    
    portion = round(len(repository) / folds)
    
    for fold, train_minimal, test_minimal in zip(range(folds), train_minimals, test_minimals):
        lower_limit = fold * portion
        upper_limit = (fold + 1) * portion
        
        random_sample = repository[lower_limit:upper_limit]
        movie_lens_validation = pd.concat([test_minimal, random_sample])
        
        movie_lens_train = repository.drop(index=random_sample.index)
        movie_lens_train = pd.concat([movie_lens_train, train_minimal])

        movie_lens_train_matrix = movie_lens_train.pivot(index='userId', columns='movieId', values='rating')

        yield movie_lens_train_matrix, movie_lens_validation
```

## Test Cross Validation


```python
train_datasets = []
test_datasets = []
counter = 0

progress_bar = tqdm(total=folds)

for cross_validation_train, cross_validation_test in generate_cross_validated_sample(movie_lens_train, seed=random_seed, folds=folds):
    # Train Datasets do not repeat completely
    assert not any(cross_validation_train.isin(train_dataset).all().any() for train_dataset in train_datasets)
    
    # Test Datasets do not repeat
    assert not any(cross_validation_test.isin(test_dataset).any().any() for test_dataset in test_datasets)
    
    # Train and Test Datasets do not share entries
    assert not cross_validation_train.isin(cross_validation_test).any().any()
    
    # Train have the same shape as original
    assert cross_validation_train.shape == (max_user_limits, max_movie_limits)
    
    train_datasets.append(cross_validation_train)
    test_datasets.append(cross_validation_test)
    counter += 1
    progress_bar.update(1)

progress_bar.close()

assert counter == folds
```

# Stochastic Mini-Batch Gradient Descent with L2 Regularization

## SVD Approximation with Gradient Descent

According to SVD, $X = U \Sigma V$. Assuming $\Sigma$ merges into $U$ or $V^T$, the matrix factorization becomes $X = U V^T$.

Let $r_{ui}$ be the rating of user $u$ for item $i$. Because of the way a matrix product is defined, the value of $r_{ui}$ is the result of a dot product between two vectors: a vector $p_u$, which is a row of $U$ and is specific to the user $u$, and a vector $q_i$, which is a column of $V^T$ and is specific to the item $i$:
 
$$\begin{pmatrix} &  & \\ & r_{ui} & \\ &  & \end{pmatrix} = \begin{pmatrix} . \\ \cdots p_u \cdots \\ . \end{pmatrix} \begin{pmatrix} & \vdots  & \\ . & q_i & . \\  & \vdots & \end{pmatrix}$$

$$r_{ui} = p_u \cdot q_i$$

The vector $p_u$ and $q_i$ represent the affinity of user $u$ and item $i$ for each of the latent factors in $U$ and $V^T$, respectively.

When a matrix $X$ has missing entries, the SVD of $X$ is not defined, the matrices $XX^T$ and $X^TX$ do not exist, so their eigenvectors do not exist either, and $X$ cannot be factorized as $U \Sigma V^T$. A simple solution is to fill the missing entries of $X$ with some heuristic value, like the mean of the columns of rows. However, the results are usually highly biased, and it is importante to note that the missing values are feedback that the user did not give, and therefore should not be considered negative or null rating. Luckily, computing the eigenvectors of $XX^T$ and $X^TX$ is not the only way of computing the SVD of $X$. The matrices $U$ and $V^T$ can be found if the vectors $p_u$ and $q_i$ ($p_u$ make up the rows of $U$ and $q_i$ make up the columns of $V^T$) are found such that:

* $r_{ui} = p_u \cdot q_i \quad \forall \; u,i$
* All the vectors $p_u$ are mutually orthogonal, as well as the vectors $q_i$

It is sought to match as well as possible the values $r_{ui}$ with what they are supposed to be, $p_u \cdot q_i$, making the error minimal:

$\min_{p_u, q_i \\ p_u \perp p_v \\ q_i \perp q_j} \sum_{r_{ui} \in X} (r_{ui} - p_u \cdot q_i)^2$

As $X$ is incomplete, the missing entries are ignored, as well as the orthogonality constraints, resulting in the following optimization problem, proposed by Simon Funk:

$\min_{p_u, q_i}\sum_{r_{ui} \in X} (r_{ui} - p_u \cdot q_i)^2.$

As this optimization problem is not convex, it will be very difficult to find the values of the vectors $p_u$ and $q_i$ that make this sum minimal, and the optimal solution may not even be unique. Therefore, Gradient Descent is used. The function to be minimized is then:

$f(p_*, q_*) = \sum_{r_{ui} \in X} (r_{ui} - p_u \cdot q_i)^2 =\sum_{r_{ui} \in X} f_{ui}(p_u, q_i)$

with its both partial derivatives being:

$\frac{\partial f_{ui}}{\partial p_u} = - 2 q_i (r_{ui} - p_u \cdot q_i)$

$\frac{\partial f_{ui}}{\partial q_i} = - 2 p_u (r_{ui} - p_u \cdot q_i)$

**L2 regularization**

$\underset{w}{\min} f(w) + \lambda ||w||_2^2$


```python
def l2_regularized_mini_batch_stochastic_gradient_descent(X_, alpha=0.01, lambda_=0.5, num_epochs=50, mini_batch_size=0, r=None, seed=42):
    
    # Pandas DataFrame to NumPy
    X = X_.to_numpy()

    if r is None:
        r = X.shape[1]
    
    np.random.seed(seed)
    
    # Initialize p and q
    variance = 0.1    
    p = np.random.normal(0, variance, (X.shape[0], r))
    q = np.random.normal(0, variance, (X.shape[1], r))
    
    train_rmses = []
    
    # Iterate only over non-nan indexes
    non_nan_indexes = list(zip(*np.where(~np.isnan(X))))
    
    if mini_batch_size == 0:
        mini_batch_size = len(non_nan_indexes) - 1
    
    for _ in range(num_epochs):
        
        # Stochastic part
        np.random.shuffle(non_nan_indexes)
        
        # Save p and q
        p_new = p.copy()
        q_new = q.copy()
        
        for index, (u, i) in enumerate(non_nan_indexes):
            
            # Update p and q every mini_batch_size iterations
            if index % mini_batch_size == 0:
                p = p_new.copy()
                q = q_new.copy()
            
            # Update p and q new with L2 regularization derivatives
            error = X[u][i] - np.dot(p[u], q[i]) 
            
            p_new[u] += alpha * (error * q[i] - lambda_ * (2 * p[u]))
            q_new[i] += alpha * (error * p[u] - lambda_ * (2 * q[i]))
        
        # Calculate RSME
        rmse = np.sqrt(np.nanmean((X - p @ q.T) ** 2))
        train_rmses.append(rmse)
        
    return p, q, train_rmses
```


```python
def calc_rmse(train, test, p, q):
    errors = []
    
    X_approx = p @ q.T
    
    for _, (user_, movie_, rating) in test[["userId", "movieId", "rating"]].iterrows():
        user_index = np.argmax(train.index == user_)
        movie_index = np.argmax(train.columns == movie_)
        error = rating - X_approx[user_index, movie_index]
        errors.append(error)

    errors = np.array(errors)
    return np.sqrt(np.mean(errors ** 2))
```

## Lambda Optimization


```python
# Calculate Train and Test RSME for each Lambda

train_rmseses_means = []
test_rmseses_means = []
errors = []

for lambda_ in tqdm(lambdas):
    cross_validation_generator = generate_cross_validated_sample(movie_lens_train, seed=random_seed, folds=folds)
    
    train_rmseses = []
    test_rmseses = []
    
    for fold, (cross_validation_train_matrix, cross_validation_test) in enumerate(cross_validation_generator, start=1):
        p, q, train_rmse = l2_regularized_mini_batch_stochastic_gradient_descent(cross_validation_train_matrix, seed=random_seed, 
                                                                     lambda_=lambda_, num_epochs=num_epochs, mini_batch_size=1, alpha=alpha)

        train_rmseses.append(train_rmse[-1])
        
        test_rmse = calc_rmse(cross_validation_train_matrix, cross_validation_test, p, q)

        test_rmseses.append(test_rmse)

        data_point = (fold, test_rmse, lambda_, *cross_validation_train_matrix.shape)
        errors.append(data_point)
        
    train_rmseses_means.append(np.mean(train_rmseses))
    test_rmseses_means.append(np.mean(test_rmseses))
```

```python
# Results for cross-validated training dataset

best_iteration = np.argmin(test_rmseses_means)

cross_validation_rmse_df = pd.DataFrame(columns=["Fold", "Test RMSE", "Lambda", "UserSize", "MovieSize"], data=errors)
cross_validation_rmse_df["Lambda"] = np.round(cross_validation_rmse_df["Lambda"], 2)

cross_validation_rmse_train = train_rmseses_means[best_iteration]
cross_validation_rmse_test = test_rmseses_means[best_iteration]

best_lambda = lambdas[best_iteration]

print(f'{folds}-Folds Cross-validation results:')
print(f"Best Lambda over {folds}-Folds: {best_lambda:.2f}")
print(f'Train RMSE: {cross_validation_rmse_train:.4f}')
print(f'Test RMSE: {cross_validation_rmse_test:.4f}\n')

# Using whole dataset (train and validation)

p, q, rmse_train = l2_regularized_mini_batch_stochastic_gradient_descent(movie_lens_train_matrix, seed=random_seed, lambda_=best_lambda, 
                                                             num_epochs=num_epochs, mini_batch_size=1, alpha=alpha)
rmse_validation = calc_rmse(movie_lens_train_matrix, movie_lens_validation, p, q)

print('Whole data set results:')
print(f"Train RMSE: {rmse_train[-1]:.4f}")
print(f'Validation RMSE: {rmse_validation:.4f}')
```

    5-Folds Cross-validation results:
    Best Lambda over 5-Folds: 0.09
    Train RMSE: 0.6199
    Test RMSE: 0.8953
    
    Whole data set results:
    Train RMSE: 0.7362
    Validation RMSE: 0.8643
    


```python
plt.plot(rmse_train)
```


    
![image alt text]({static}../images/collaborative_filtering_5.png)
    



```python
sns.pointplot(x="Lambda", y="Test RMSE", data=cross_validation_rmse_df, scale=0.35, errwidth=1, capsize=0.05, ci=99)
```


    
![image alt text]({static}../images/collaborative_filtering_1.png)
    


## Mini-Batch Size Optimization


```python
# Calculate Train and Test RSME for each Mini-Batch Size

train_rmseses_means = []
test_rmseses_means = []
errors = []

for size in tqdm(mini_batch_sizes):
    cross_validation_generator = generate_cross_validated_sample(movie_lens_train, seed=random_seed, folds=folds)
    
    train_rmseses = []
    test_rmseses = []
    
    for fold, (cross_validation_train_matrix, cross_validation_test) in enumerate(cross_validation_generator, start=1):
        p, q, train_rmse = l2_regularized_mini_batch_stochastic_gradient_descent(cross_validation_train_matrix, seed=random_seed, 
                                                                     lambda_=best_lambda, num_epochs=num_epochs, mini_batch_size=size, alpha=alpha)

        train_rmseses.append(train_rmse[-1])
        
        test_rmse = calc_rmse(cross_validation_train_matrix, cross_validation_test, p, q)

        test_rmseses.append(test_rmse)

        data_point = (fold, test_rmse, size, *cross_validation_train_matrix.shape)
        errors.append(data_point)
        
    train_rmseses_means.append(np.mean(train_rmseses))
    test_rmseses_means.append(np.mean(test_rmseses))
```

```python
# Results for cross-validated training dataset

best_iteration = np.argmin(test_rmseses_means)

cross_validation_rmse_df = pd.DataFrame(columns=["Fold", "Test RMSE", "Mini-Batch-Size", "UserSize", "MovieSize"], data=errors)
cross_validation_rmse_train = train_rmseses_means[best_iteration]
cross_validation_rmse_test = test_rmseses_means[best_iteration]

best_size = mini_batch_sizes[best_iteration]

print(f'{folds}-Folds Cross-validation results:')
print(f"Best Mini-Batch-Size over {folds}-Folds: {best_size:.2f}")
print(f'Train RMSE: {cross_validation_rmse_train:.4f}')
print(f'Test RMSE: {cross_validation_rmse_test:.4f}\n')

# Using whole dataset (train and validation)

p, q, rmse_train = l2_regularized_mini_batch_stochastic_gradient_descent(movie_lens_train_matrix, seed=random_seed, lambda_=best_lambda, 
                                                             num_epochs=num_epochs, mini_batch_size=best_size, alpha=alpha)
rmse_validation = calc_rmse(movie_lens_train_matrix, movie_lens_validation, p, q)

print('Whole data set results:')
print(f"Train RMSE: {rmse_train[-1]:.4f}")
print(f'Validation RMSE: {rmse_validation:.4f}')
```

    5-Folds Cross-validation results:
    Best Mini-Batch-Size over 5-Folds: 128.00
    Train RMSE: 0.6203
    Test RMSE: 0.8946
    
    Whole data set results:
    Train RMSE: 0.7359
    Validation RMSE: 0.8619
    


```python
plt.plot(rmse_train)
```


    
![image alt text]({static}../images/collaborative_filtering_2.png)
    

```python
sns.pointplot(x="Mini-Batch-Size", y="Test RMSE", data=cross_validation_rmse_df, scale=0.35, errwidth=1, capsize=0.05, ci=99)
```


    
![image alt text]({static}../images/collaborative_filtering_3.png)
    


# Confidence Intervals for RMSE with optimum parameters


```python
final_rmse_validation = []

for seed in tqdm(range(30)):
    
    movies_by_count = movie_lens.groupby("movieId", as_index=False).size().sort_values(by="size", ascending=False)["movieId"].to_numpy()
    movies_available = movies_by_count[:max_movie_limits]
    movie_lens = movie_lens[movie_lens['movieId'].isin(movies_available)]

    users_by_count = movie_lens.groupby("userId", as_index=False).size().sort_values(by="size", ascending=False)["userId"].to_numpy()
    users_available = users_by_count[:max_user_limits]
    movie_lens = movie_lens[movie_lens['userId'].isin(users_available)]

    movie_lens = movie_lens.sample(frac=1, random_state=seed).reset_index(drop=True)


    train_unique_user_rows = movie_lens.groupby('userId', group_keys=False).apply(lambda x: x.sample(1, random_state=seed))

    train_unique_movie_rows = movie_lens.groupby('movieId', group_keys=False).apply(lambda x: x.sample(1, random_state=seed))
    train_unique_movie_rows = train_unique_movie_rows[~ train_unique_movie_rows["movieId"].isin(train_unique_user_rows["movieId"])]

    train_minimal = pd.concat([train_unique_movie_rows, train_unique_user_rows])


    test_base = movie_lens.drop(index=train_minimal.index)
    test_unique_user_rows = test_base.groupby('userId', group_keys=False).apply(lambda x: x.sample(1, random_state=seed))

    test_unique_movie_rows = test_base.groupby('movieId', group_keys=False).apply(lambda x: x.sample(1, random_state=seed))
    test_unique_movie_rows = test_unique_movie_rows[~ test_unique_movie_rows["movieId"].isin(test_unique_user_rows["movieId"])]

    test_minimal = pd.concat([test_unique_movie_rows, test_unique_user_rows])



    non_used_rows = movie_lens.drop(index=train_minimal.index).drop(index=test_minimal.index)

    random_sample = non_used_rows.sample(frac=0.9, random_state=seed)

    movie_lens_train = pd.concat([train_minimal, random_sample])
    movie_lens_validation = movie_lens.drop(index=movie_lens_train.index)


    movie_lens_train_matrix = movie_lens_train.pivot(index='userId', columns='movieId', values='rating')
    
    p, q, rmse_train = l2_regularized_mini_batch_stochastic_gradient_descent(movie_lens_train_matrix, seed=seed, lambda_=best_lambda, 
                                                      num_epochs=num_epochs, mini_batch_size=best_size, alpha=alpha)
    
    rmse_validation = calc_rmse(movie_lens_train_matrix, movie_lens_validation, p, q)
    final_rmse_validation.append(rmse_validation)
```

```python
df = pd.DataFrame(columns=["RMSE"], data=final_rmse_validation)
sns.pointplot(x="RMSE", data=df, capsize=0.2, orient="h", color='saddlebrown')
```

    
![image alt text]({static}../images/collaborative_filtering_4.png)
    


# Predicted matrix


```python
lambda_ = 0.09
mini_batch_size = 128
p, q, rmse_train = l2_regularized_mini_batch_stochastic_gradient_descent(movie_lens_train_matrix, seed=random_seed, lambda_=lambda_, 
                                                      num_epochs=num_epochs, mini_batch_size=mini_batch_size, alpha=alpha)

X_approx = pd.DataFrame(np.round(p @ q.T / 0.5) * 0.5)
X_approx.columns = movie_lens_train_matrix.columns
X_approx.index = movie_lens_train_matrix.index
```


```python
movie_df = pd.read_csv(r"C:\Users\Gastón\Office\Data Science\_Datasets\movie_lens_100k_movies.csv", sep='|', names=['movie id', 'movie title', 'release date',
                                                                                                                        'video release date', 'IMDb URL', 'unknown',
                                                                                                                        'Action', 'Adventure', 'Animation', 'Children',
                                                                                                                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                                                                                                        'FilmNoir', 'Horror', 'Musical', 'Mystery', 'Romance',
                                                                                                                        'SciFi', 'Thriller', 'War', 'Western'])

movie_df = movie_df[['movie id','movie title']]
movie_df = movie_df.set_index('movie id', drop=True)
```


```python
def recommend_movie(user, num_movies):
    recommended_movies_id = X_approx.loc[user].sort_values(ascending=False)[:num_movies].index.to_numpy()
    recommended_movies = movie_df.loc[recommended_movies_id].set_index('movie title')
    return recommended_movies
```


```python
user = X_approx.sample().index[0]
num_movies = 10
recommended_movies = recommend_movie(user, num_movies)
print(f'Recommended movies for user {user}:')
recommended_movies
```

    Recommended movies for user 416:
    




<div>
<table border="1">
  <tbody>
    <tr>
      <th>Godfather, The (1972)</th>
    </tr>
    <tr>
      <th>Schindler's List (1993)</th>
    </tr>
    <tr>
      <th>Shawshank Redemption, The (1994)</th>
    </tr>
    <tr>
      <th>Raiders of the Lost Ark (1981)</th>
    </tr>
    <tr>
      <th>Princess Bride, The (1987)</th>
    </tr>
    <tr>
      <th>Star Wars (1977)</th>
    </tr>
    <tr>
      <th>Rear Window (1954)</th>
    </tr>
    <tr>
      <th>Braveheart (1995)</th>
    </tr>
    <tr>
      <th>Titanic (1997)</th>
    </tr>
    <tr>
      <th>Lawrence of Arabia (1962)</th>
    </tr>
  </tbody>
</table>
</div>


<hr>

# References

[Collaborative Filtering - Wikipedia](https://www.wikiwand.com/en/Collaborative_filtering){: target="_blank"}

[Build a Recommendation Engine With Collaborative Filtering](https://realpython.com/build-recommendation-engine-collaborative-filtering/#how-to-find-similar-users-on-the-basis-of-ratings){: target="_blank"}
