"""
Semi - Supervised Kernel K - Means
"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels 


class KernelKMeans(object):
    def __init__(self, k, kernel='rbf', gamma=None, known_data=None, coef0=0, deg=None, max_iter=100, alpha=0.5, verbose=False):
        """
            :param k: The number of clusters
            :param metric: The kernel matrix to compute. Valid values are:  
                "rbf", "sigmoid", "polynomial", "poly", "linear", "cosine"
                "euclidean" (default), "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis", "linear"
            :param known_data: A 2D array of indexes of known points. When using this parameter, every cluster that  
                does not contains known labels should be represented as empty list. For example, if we have a dataset and
                we know some points for classes 1 and 3, we would have  
                ```known_data=[[1,2,3,4], [], [19,20,21]]```
            :param coef0: The coefficient of the different kernels (default: 0)
            :param gamma: The gamma value of rbf and sigmoid kernel
            :param deg: The degree for the polynomial kernel
            :param max_iter: The max number of iterations if convergance did not reach (default: 100)
            :param alpha: When using semi supervised clustering, we can weigh the known data points differently. 
                The range of this paremeter is between 0 < \alpha < 1. 
            :param verbose: Prints iterations and convergence rate when set to True.

            :type k: int
            :type metric: str
            :type coef0: float
            :type gamma: float
            :type deg: float
            :type known_data: np.array
            :type max_iter: int
            :type alpha: float
            :type verbose: bool
            
        """
        self.k = k
        self.kernel_params_ = {'coef0':coef0, 'gamma':gamma, 'degree':deg}
        self.kernel_ = kernel
        self.kernel_distance = None
        self.max_iter = max_iter
        if known_data is not None:
            self.known_data = np.array(known_data)
        else:
            self.known_data = None

        self.alpha = float(alpha)

        self.verbose = verbose
        self.weights = None

        
  

    def _get_kernel(self, data):
        """
            Sets the kernel matrix
            :param data: The data to cluster
            :type data: np.array
        """
        try:
            self.dist = pairwise_kernels(data, metric=self.kernel_, filter_params=True, **self.kernel_params_)
        except Exception as e:
            print e
            return



    def fit(self, data):
        """
            A helper function that decides if to use semi supervised or unsupervised clustering

            :param data: The data to cluster
            :type data: np.array 
        """
        
        self._get_kernel(data)        
        
        if self.known_data is not None:
            self._fit_biased(data)
        else:
            self._fit(data)


    def _fit(self, data):
        """
            Perfom unsupervised clustering

            :param data: The data to cluster
            :type data: np.array
        """

        labels = np.random.choice(np.arange(0, self.k), data.shape[0])
        first_term  = self.dist[np.arange(len(data)), np.arange(len(data))]
        current_iter = 0
        labels_changed = np.infty
        while current_iter<self.max_iter and labels_changed>0:

            temp = np.zeros((self.k, len(data)))

            for i in xrange(self.k):
                inds = np.where(labels==i)[0]
                second_term = (-2 * self.dist[:,inds].sum(axis=1))/len(inds)
                third_term = (self.dist[inds][:,inds]).sum()/(len(inds)**2)

                temp[i] = first_term + second_term + third_term

            old_labels = labels
            labels = np.argmin(temp,axis=0)
            labels_changed = (labels!=old_labels).sum()
            
            if self.verbose:
                print "Iteration {} of {}, {} labels changed".format(current_iter+1, self.max_iter, labels_changed)
            current_iter+=1

        self.labels_ = labels




    def _fit_biased(self, data):
        """
            Perfom semi-unsupervised clustering

            :param data: The data to cluster
            :type data: np.array
        """
        
        # Assign random labels
        labels = np.random.choice(np.arange(0, self.k), data.shape[0])
        
        # Put labels for the known data
        for i in xrange(self.k):
            if self.known_data[i] is not None and len(self.known_data[i]):
                labels[self.known_data[i]] = i

     
        # Compute first term
        first_term  = self.dist[np.arange(len(data)), np.arange(len(data))] 
        

        current_iter = 0
        labels_changed = np.infty
        weights = np.zeros(data.shape[0])

        while current_iter<self.max_iter and labels_changed>0:

            temp = np.zeros((self.k, len(data)))


            
            for i in xrange(self.k):

                # compute weights for every cluster
                
                weights.fill(self.alpha)
                weights[self.known_data[i]] = 1-self.alpha
                weights = weights/weights.sum()   
                
                # Find indexes
                inds = np.where(labels==i)[0]

                # Compute second and third term
                second_term = ((-2 * self.dist[:,inds].sum(axis=1))/len(inds)) 
                third_term = (self.dist[inds][:,inds]).sum()/(len(inds)**2)

            
                # Final result
                temp[i] = (first_term + second_term + third_term)*weights


            # Check how many labels changed (Stopping condition)
            old_labels = labels
            labels = np.argmin(temp,axis=0)
            labels_changed = (labels!=old_labels).sum()
            
            
            if self.verbose:
                print "Iteration {} of {}, {} labels changed".format(current_iter+1, self.max_iter, labels_changed)
            
            current_iter+=1


        self.labels_ = labels


    def predict(self):
        """
            :return: The labels of the clustered data
            :rtype: np.array
        """
        return self.labels_


    def fit_predict(self, data):
        """
            Clusters the data and return the labels
            :param data: The data to cluster
            :type data: np.array
        """
        self.fit(data)
        return self.predict()
