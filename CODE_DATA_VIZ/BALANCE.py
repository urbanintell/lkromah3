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
from sklearn.decomposition import FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from evaluate import evaluate_kmeans_for_labeled_data
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import mixture
import numpy as np 
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis
from evaluate import evaluate_kmeans
from evaluate import evaluate_em
from evaluate import nn_learner
from viz import cluster_viz_km
# Read in the data. Randomize and take a sample of 10,000
balance = pandas.read_csv(
    "/Users/lusenii/Google Drive/Assignment_3/data/balance.csv",header=None,low_memory=False)

#true labels
true_labels = balance.iloc[:,0]

cluster_viz_km(balance,6,'')

#number of values of k 
n = 20

# Kmeans 

# ofile = open('balance_kmeans.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'adjusted_rand_score', 'homogeneity_score','log-likehood','algo'])

# print('Fitting Kmeans model..')

for num in range(2, n):
    evaluate_kmeans_for_labeled_data(KMeans(init='k-means++', n_clusters=num, n_init=10),
                  cluster=num, data=balance, true_labels=true_labels,algo='Kmeans',writer=writer)


print('Kmeans Complete')
ofile.close()

# ofile = open('balance_em.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'Log-likelihood','bic','aic','algo'])

# print('Fitting EM model..')
# train, test = train_test_split(balance, test_size = 0.2)
# # EM
# for num in range(2, n):

#     evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
#                   cluster=num, data=train,algo='EM',writer=writer,test=test)

# print('EM Complete')

# ofile.close()


# ## Dimensionality Reduction
reduced_to = 2
# PCA
#----------------------------------------
# print('Fitting PCA model..')
# ofile = open('balance_kmeans_pca.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'adjusted_rand_score', 'homogeneity_score','log-likelihood','algo'])

# reduced_data = PCA(n_components=reduced_to).fit_transform(balance)
# cluster_viz_km(reduced_data,6,'PCA')

# print('Fitting Kmeans model..')
# #Kmeans
# for num in range(2, n):
#     evaluate_kmeans_for_labeled_data(KMeans(init='k-means++', n_clusters=num, n_init=10),
#                   cluster=num, data=reduced_data, true_labels=true_labels,algo='Kmeans_PCA',writer=writer)


# ofile.close()
# print('Kmeans closed')

# ofile = open('balance_em_pca.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'bic','aic','algo'])
# train, test = train_test_split(balance, test_size = 0.2)
# print('Fitting EM model..')
# # EM
# for num in range(2, n):
#     evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
#                   cluster=num, data=train,algo='EM_PCA', writer=writer,test=test)

# print('EM Complete')
# print('PCA Complete')
# ofile.close()


# ICA
# -----------------------------------
print('Fitting ICA model..')
ofile = open('balance_kmeans_ica.csv', "w")
writer = csv.writer(ofile, delimiter=',')
writer.writerow(['k', 'time', 'adjusted_rand_score', 'homogeneity_score','log-likelihood','algo'])

reduced_data = FastICA(n_components=reduced_to,algorithm='parallel').fit_transform(balance)
# cluster_viz_km(reduced_data,6,'ICA')
print (kurtosis(reduced_data))


#KMeans
for num in range(2, n):
    evaluate_kmeans_for_labeled_data(KMeans(init='k-means++', n_clusters=num, n_init=10),
                  cluster=num, data=reduced_data, true_labels=true_labels,algo='Kmeans_ICA',writer=writer)

print('Completed KM ICA model')
ofile.close()

ofile = open('balance_em_ica.csv', "w")
writer = csv.writer(ofile, delimiter=',')
writer.writerow(['k', 'time', 'Log-likelihood','bic','aic','algo'])

print('Fitting EM model..')
train, test = train_test_split(balance, test_size = 0.2)
# EM
for num in range(2, n):
    evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
                  cluster=num, data=train,algo='EM_ICA',writer=writer,test=test)

print('EM ICA Complete')

ofile.close()

#Randomized Projections
#------------------------------------
# print('Fitting Randomized Projection model..')
# ofile = open('balance_kmeans_randomized_projections.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'adjusted_rand_score', 'homogeneity_score','log-likelihood','algo'])

# reduced_data = GaussianRandomProjection(n_components=reduced_to, eps=0.1, random_state=None).fit_transform(balance)
# # cluster_viz_km(reduced_data,6,'Randomized Projection')
# #KMeans
# print ('Kmeans ')
# for num in range(2, n):
#     evaluate_kmeans_for_labeled_data( KMeans(init='k-means++', n_clusters=num, n_init=10),
#                   cluster=num, data=reduced_data, true_labels=true_labels, algo='Kmeans_RP',writer=writer)

# print('Completed Kmeans Randomized Project model')
# ofile.close()

# ofile = open('balance_em_randomized_projections.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'Log-likelihood','bic','aic','algo'])

# print('Fitting EM model..')
# train, test = train_test_split(balance, test_size = 0.2)
# # EM
# for num in range(2, n):
#     evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
#                   cluster=num, data=train,algo='EM_RP',writer=writer,test=test)

# print('EM Complete')

# ofile.close()


# # # FA
# # #-----------------------------------------
# print('Fitting Factor Analysis model..')
# ofile = open('balance_kmeans_FA.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'adjusted_rand_score', 'homogeneity_score','algo'])
# reduced_data = FactorAnalysis(n_components=reduced_to, svd_method='randomized').fit_transform(balance)
# cluster_viz_km(reduced_data,6,'Factor Analysis')
# #KMeans
# for num in range(2, n):
#     evaluate_kmeans_for_labeled_data( KMeans(init='k-means++', n_clusters=num, n_init=10),
#                   cluster=num, data=reduced_data, true_labels=true_labels,algo='KM_FA',writer=writer)

# print('Completed FA model')
# ofile.close()

# ofile = open('balance_em_FA.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'Log-likelihood','bic','aic','algo'])

# print('Fitting EM model..')
# train, test = train_test_split(balance, test_size = 0.2)
# # EM
# for num in range(2, n):
#     evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
#               cluster=num, data=train,algo='EM_FA',writer=writer,test=test)

# print('EM Complete')

# ofile.close()

# NN - FA
reduced_data = FactorAnalysis(n_components=reduced_to, svd_method='randomized').fit_transform(balance)
train = reduced_data[0:500]
test = reduced_data[500:len(reduced_data)]
train_labels = balance.iloc[0:499,:].iloc[:,0]
test_labels = balance.iloc[500:len(reduced_data)-1,:].iloc[:,0]

nn_learner(train=train[1:], test=test[1:], train_target=train_labels, test_target=test_labels,red_algo='FA')

# NN - PCA
reduced_data = PCA(n_components=reduced_to).fit_transform(balance)
train = reduced_data[0:500]
test = reduced_data[500:len(reduced_data)]
train_labels = balance.iloc[0:499,:].iloc[:,0]
test_labels = balance.iloc[500:len(reduced_data)-1,:].iloc[:,0]

nn_learner(train=train[1:], test=test[1:], train_target=train_labels, test_target=test_labels,red_algo='PCA')

# NN - ICA
reduced_data = FastICA(n_components=reduced_to,algorithm='parallel').fit_transform(balance)
train = reduced_data[0:500]
test = reduced_data[500:len(reduced_data)]
train_labels = balance.iloc[0:499,:].iloc[:,0]
test_labels = balance.iloc[500:len(reduced_data)-1,:].iloc[:,0]

nn_learner(train=train[1:], test=test[1:], train_target=train_labels, test_target=test_labels,red_algo='ICA')


# NN - RP
reduced_data = GaussianRandomProjection(n_components=reduced_to, eps=0.1, random_state=None).fit_transform(balance)
train = reduced_data[0:500]
test = reduced_data[500:len(reduced_data)]
train_labels = balance.iloc[0:499,:].iloc[:,0]
test_labels = balance.iloc[500:len(reduced_data)-1,:].iloc[:,0]

nn_learner(train=train[1:], test=test[1:], train_target=train_labels, test_target=test_labels,red_algo='RP')