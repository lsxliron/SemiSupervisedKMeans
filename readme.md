# Semi Supervised K-Means and Kernel K-Means Clustering

### Installation
- Clone the repository
```
pip install -r requirements.txt
```

### Usage:
You can run the exmple `python example.py`

** Full documentation can be found in the `docs` folder**
---
# API

## KMeans
Semi - Supervised K - Means

class **kmeans.KMeans**(k, threshold=0.0001, max_iter=30, metric='euclidean', known_data=None, alpha=1, verbose=False)

Bases: "object"

Parameters:
* **k** (*int*) -- The number of clusters

  * **threshold** (*float*) -- The convergance threshold
    (default: 0.0001)

  * **max_iter** (*int*) -- The max number of iterations if
    convergance did not reach (default: 30)

  * **metric** (*str*) -- The distance metric to use. Valid
    values are:   "euclidean" (default), "manhattan", "chebyshev",
    "minkowski", "wminkowski", "seuclidean", "mahalanobis"

  * **known_data** -- A 2D array of indexes of known points.
    When using this parameter, every cluster that   does not
    contains known labels should be represented as empty list. For
    example, if we have a dataset and we know some points for
    classes 1 and 3, we would have
    "`known_data=[np.array([1,2,3,4]), np.array([]),
    np.array([19,20,21])]`"

  * **alpha** (*float*) -- When using semi supervised
    clustering, we can weigh the known data points differently
    (default: 1) alpha=1 is equivalent to unsupervised clustering.

  * **verbose** (*bool*) -- Prints iterations and convergence
    rate when set to True (default: False)

    **fit(data)**

        Clusters the data

      Parameters:
         **data** (*np.ndarray*) -- The data to cluster

    **fit_predict(data)**  
    Clusteres the data and returns tha labels
      
      Parameters:
        **data** (*np.ndarray*) -- The data to cluster

      Returns:
        The data labels

      Return type:
        np.array

    **predict()**

    Returns:
        The labels of the clustered data

    Return type:
        np.array

---


## Kernel K Means

Semi - Supervised Kernel K - Means

class **kernelkmeans.KernelKMeans**(k, kernel='rbf', gamma=None, known_data=None, coef0=0, deg=None, max_iter=100, alpha=0.5, verbose=False)

Bases: "object"

Parameters:
  * **k** (*int*) -- The number of clusters

  * **metric** (*str*) -- The kernel matrix to compute. Valid
    values are:   "rbf", "sigmoid", "polynomial", "poly",
    "linear", "cosine" "euclidean" (default), "manhattan",
    "chebyshev", "minkowski", "wminkowski", "seuclidean",
    "mahalanobis", "linear"

  * **known_data** (*np.array*) -- A 2D array of indexes of
    known points. When using this parameter, every cluster that
    does not contains known labels should be represented as empty
    list. For example, if we have a dataset and we know some
    points for classes 1 and 3, we would have
    "`known_data=[[1,2,3,4], [], [19,20,21]]`"

  * **coef0** (*float*) -- The coefficient of the different
    kernels (default: 0)

  * **gamma** (*float*) -- The gamma value of rbf and sigmoid
    kernel

  * **deg** (*float*) -- The degree for the polynomial kernel

  * **max_iter** (*int*) -- The max number of iterations if
    convergance did not reach (default: 100)

  * **alpha** (*float*) -- When using semi supervised
    clustering, we can weigh the known data points differently.
    The range of this paremeter is between 0 < lpha < 1.

  * **verbose** (*bool*) -- Prints iterations and convergence
    rate when set to True.

    **fit(data)**

      A helper function that decides if to use semi supervised or
      unsupervised clustering

      Parameters:
             **data** (*np.array*) -- The data to cluster

    **fit_predict(data)**

      Clusters the data and return the labels :param data: The data to
      cluster :type data: np.array

     **predict()**

      Returns:
         The labels of the clustered data

      Return type:
         np.array
