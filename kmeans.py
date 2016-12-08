"""
Semi - Supervised K - Means
"""
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

class KMeans(object):

    def __init__(self, k, threshold=0.0001, max_iter=30, metric='euclidean', known_data=None, alpha=0.5, verbose=False):
        """
            :param k: The number of clusters
            :param threshold: The convergance threshold (default: 0.0001)
            :param max_iter: The max number of iterations if convergance did not reach (default: 30)
            :param metric: The distance metric to use. Valid values are:  
                "euclidean" (default), "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis" 
            :param known_data: A 2D array of indexes of known points. When using this parameter, every cluster that  
                does not contains known labels should be represented as empty list. For example, if we have a dataset and
                we know some points for classes 1 and 3, we would have  
                ```known_data=[np.array([1,2,3,4]), np.array([]), np.array([19,20,21])]```
            :param alpha: When using semi supervised
                    clustering, we can weigh the known data points differently.
                    The range of this paremeter is between 0 <= alpha <= 1.
            :param verbose: Prints iterations and convergence rate when set to True (default: False)

            :type k: int
            :type threshold: float
            :type max_iter: int
            :type metric: str
            :type know_data: list
            :type alpha: float
            :type verbose: bool
        """
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter
        self.centroids = None
        self.labels_  = None
        self.metric = metric
        self.known_data = known_data
        self.alpha = float(alpha)
        self.verbose = verbose

    def _validate_metric(self):
        """
            Validates that the metric parameter
        """
        try:
            pairwise(np.array([0,0]).reshape(-1,1), np.array([1,1]).reshape(-1,1), metric=self.metric)
        except Exception as e:
            print e
            return

    def _get_distance(self, x, y, reshape=True):
        """
            :param x: point 1
            :param y: point 2
            :type x: np.array
            :type y: np.array
            :return: The distance between point 1 and point 2
            :rtype: float
        """
        if not reshape:
            return pairwise_distances(x, y, metric=self.metric)[0]

        return pairwise_distances(x.reshape(1,-1), y, metric=self.metric)[0]



    def _update_centroids(self, data):
        """
            Updates the clusters centroids by taking the mean of all the points the belongs ot that cluster
            :param data: The data to cluster
            :type data: np.array
        """
        for i in xrange(self.k):
            self.centroids[i] = np.mean(data[np.where(self.labels_==i)], axis=0)



            
    def _update_biased_centroids(self, data):
        """
            Updates the clusters centroids in the semi-supervised settings
            :param data: The data to cluster
            :type data: np.array
        """
        weights = np.zeros(data.shape[0])   
        weights.fill(1-self.alpha)     
        kd = np.hstack(np.array(self.known_data).flat).astype(np.int)
        weights[kd] = self.alpha
        weights = weights/weights.sum()   
        
        for i in xrange(self.k):
        
            # compute weights for every cluster
            inds = np.where(self.labels_==i)[0]
            self.centroids[i] = np.average(data[inds], weights=weights[inds], axis=0)

        for i in xrange(self.k):
            if i<len(self.known_data) and len(self.known_data[i]):
                max_vote = 0
                max_vote = map(lambda lbl: (self.labels_[self.known_data[i]]==lbl).sum() , range(0, self.k))
                self.labels_[self.known_data[i]] = np.argmax(max_vote)


    def _kmeans_pp(self, data):
        """
            Initialize cluster centers using KMeans++

            :param data: The dataset to cluster
            :type data: np.array
        """
        
        # Fill random labels
        self.labels_ = np.random.choice(np.arange(0,self.k), len(data))
        self.labels_.fill(-1)

        # If we have some data points, make them the centroid
        # known_classes = 0
        # if self.known_data is not None:
        #     known_classes = sum(map(lambda x: 1 if len(x)>0 else 0, self.known_data))
        
        if 1<0 and self.known_data is not None:# and known_classes>1:
            current_centers = None
            for i, pts in enumerate(self.known_data):
                if len(pts):
                    self.labels_[pts] = i
                    if current_centers is not None:
                        current_centers = np.vstack((current_centers, np.mean(data[np.where(self.labels_==i)], axis=0)))
                        # current_centers = np.vstack((current_centers, data[self.known_data[i][0]]))
                    else:
                        current_centers = np.mean(data[np.where(self.labels_==i)], axis=0)
                        current_centers = current_centers.reshape(1,len(current_centers))
                        # current_centers = data[self.known_data[i][0]].reshape(1,-1)
        
        else:
            # Choose the first centroid randomly and do kmeans++
            first_centroid_index = np.random.choice(np.arange(0, len(data)), 1)
            self.labels_[first_centroid_index] = 0
            current_centers = data[first_centroid_index]


        for i in xrange(len(current_centers), self.k):
            found_centroid = False
            distances = np.array([min([np.inner(center-p,center-p) for center in current_centers]) for p in data], dtype=np.float64)
            probabilities = distances/distances.sum()
            cum_probabilities = probabilities.cumsum()

            r = np.random.rand()

            counter = 0
            while not found_centroid and counter<len(data):
                if r < cum_probabilities[counter]:
                    found_centroid = True

                else:
                    counter +=1
        
            current_centers = np.vstack((current_centers, data[counter]))
            next_label = 0
            while next_label in set(self.labels_):
                next_label += 1

            # find next label
            self.labels_[counter] = next_label


        self.centroids = current_centers


    def predict(self):
        """
            :return: The labels of the clustered data
            :rtype: np.array
        """
        return self.labels_


    def fit_predict(self, data):
        """
            Clusteres the data and returns tha labels

            :param data: The data to cluster
            :type data: np.ndarray
            :return: The data labels
            :rtype: np.array
        """
        self.fit(data)
        return self.predict()



    def fit(self, data):
        """
            Clusters the data

            :param data: The data to cluster
            :type data: np.ndarray
        """
        # Find initial centroids
        self._kmeans_pp(data)
        new_labels = self.labels_.copy()
        counter = 0
        threshold = np.infty
        
        while counter<self.max_iter and self.threshold < threshold:
            for i, p in enumerate(data):
                new_labels[i] = np.argmin(self._get_distance(p, self.centroids))
            
            old_centroids = self.centroids.copy()
            self.labels_ = new_labels.copy()

            if self.known_data is not None:
                self._update_biased_centroids(data)
            else:
                self._update_centroids(data)


            threshold = abs(np.mean(old_centroids)-np.mean(self.centroids))

            counter+=1

            if self.verbose:
                print "Iteration {}\tConvergance: {}".format(counter, threshold)
