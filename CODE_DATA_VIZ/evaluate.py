from time import time
import numpy as np
import pandas
import csv
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.neural_network import MLPClassifier
import numpy as np 

def evaluate_kmeans(estimator, cluster, data,algo, writer):
    t0 = time()
    print ('k=',cluster)
     # fit data 
    estimator.fit(data)
    #get predicted labels
    labels = estimator.labels_
    #write to file
    writer.writerow([cluster, (time() - t0), (metrics.silhouette_score(data, estimator.labels_,metric='euclidean')), (metrics.calinski_harabaz_score(data, labels)), estimator.score(data), algo])

def evaluate_em(estimator, cluster, data,algo, writer,test):
    t0 = time()
    print ('k=',cluster)
    estimator.fit(data)
    writer.writerow([cluster, (time() - t0), 
        estimator.lower_bound_, estimator.bic(test), estimator.aic(test), algo])




def evaluate_kmeans_for_labeled_data(estimator, cluster, data, true_labels,algo, writer):
    t0 = time()
    print ('k=',cluster)
     # fit data 
    estimator.fit(data)
    #get predicted labels
    predicted_labels = estimator.labels_
    #write to file
    writer.writerow([cluster, (time() - t0), (metrics.adjusted_rand_score(true_labels, predicted_labels)), (metrics.homogeneity_score(true_labels, predicted_labels)), estimator.score(data), algo])



def nn_learner(train,test, train_target, test_target,red_algo):
    ofile = open('nn_'+red_algo+'.csv', "w")
    writer = csv.writer(ofile, delimiter=',')
    writer.writerow(['layer', 'time', 'testing_accuracy','ALGO'])
    layers = [1,5,10,15,20,25,50]

    for i in layers:
        t0 = time()
        nn = MLPClassifier(hidden_layer_sizes=(i),
        learning_rate='constant', learning_rate_init=0.001, momentum=0.9).fit(train,train_target)

        writer.writerow([i, (time() - t0),100-nn.score(train,train_target),red_algo ])
    
    ofile.close()
