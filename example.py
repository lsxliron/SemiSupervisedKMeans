"""
This is an example using semi - supervised K - Means and semi - supervised Kernel K - Means
"""
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import adjusted_mutual_info_score
from kmeans import KMeans
from kernelkmeans import KernelKMeans
import os

def main():

    # Load data
    iris_data = load_iris()['data']
    iris_labels = load_iris()['target']

    cancer_data = load_breast_cancer()['data']
    cancer_labels = load_breast_cancer()['target']

    print "\n\n"

    ##################################################
    #  Unsupervised and semi - supervised K - Means  #
    ##################################################
    

    # Give 20 data points from clusters 0 and 2
    n = 20
    iris_known_data = np.array([ np.random.choice(np.where(iris_labels==0)[0], n), [], np.random.choice(np.where(iris_labels==2)[0], n) ])

    print_title("IRIS Dataset - Unsupervised K - Means")
    kmeans = KMeans(k=3, verbose=True)
    kmeans_results = kmeans.fit_predict(iris_data)
    print "\nScore:\t{}\n\n".format(adjusted_mutual_info_score(iris_labels, kmeans_results))


    print_title("IRIS Dataset - Semi - Supervised K - Means")
    kmeans_semi = KMeans(k=3, known_data=iris_known_data, alpha=0.7, verbose=True)
    kmeans_semi_results = kmeans_semi.fit_predict(iris_data)
    print "\nScore:\t{}\n\n".format(adjusted_mutual_info_score(iris_labels, kmeans_semi_results))

    

    

    #########################################################
    #  Unsupervised and semi - supervised Kernel K - Means  #
    #########################################################
    

    # Give 30 data points from clusters 0 and 2
    n = 50
    cancer_known_data = np.array([ np.random.choice(np.where(cancer_labels==0)[0], n), np.random.choice(np.where(cancer_labels==1)[0], n)])
    
    print_title("Breast Cancer Dataset - Unupervised K - Means")
    kernel_k_means = KernelKMeans(k=2, kernel='polynomial', deg=0.1, coef0=0, verbose=True)
    kernel_k_means_results = kernel_k_means.fit_predict(cancer_data)

    print "\nScore:\t{}\n\n".format(adjusted_mutual_info_score(cancer_labels, kernel_k_means_results))

    print_title("Breast Cancer Dataset - Semi - Supervised K - Means")
    kernel_k_means_semi = KernelKMeans(k=2, kernel='polynomial', deg=0.1, coef0=0, alpha=0.7, known_data=cancer_known_data, verbose=True)
    kernel_k_means_semi_results = kernel_k_means_semi.fit_predict(cancer_data)
    print "\nScore:\t{}\n\n\n".format(adjusted_mutual_info_score(cancer_labels, kernel_k_means_semi_results))





def print_title(title):
    width = int(os.popen('stty size','r').read().split()[1])
    spaces = (((width-3)-len(title))/2)-1

    print '-'*(width-1)
    print '||'+' '*spaces,title+' '*spaces+"||"
    print '-'*(width-1)


    


if __name__ == '__main__':
    main()
