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
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import mixture
import numpy as np 
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis
from evaluate import evaluate_kmeans
from evaluate import evaluate_em
from viz import cluster_viz_km
# Read in the data. Randomize and take a sample of 10,000
minst = pandas.read_csv(
    "/Users/lusenii/Google Drive/Assignment_3/data/minst_train.csv",low_memory=False).sample(frac=1)[:10000]

#split data
train, test = train_test_split(minst, test_size = 0.2)

#number of values of k 
n = 20

# # Kmeans 

# ofile = open('minst_kmeans.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score', 'calinski-harabaz score','log-likehood','algo'])

# print('Fitting Kmeans model..')

# for num in range(2, n):
#     evaluate_kmeans(KMeans(init='k-means++', n_clusters=num, n_init=10),
#                   cluster=num, data=minst,algo='Kmeans',writer=writer)

# print('Kmeans Complete')
# ofile.close()

# ofile = open('minst_em.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score','bic','aic','algo'])

# print('Fitting EM model..')
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
# ofile = open('minst_kmeans_pca.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score', 'calinski-harabaz score','log-likelihood','algo'])

# reduced_data = PCA(n_components=reduced_to).fit_transform(minst)

# print('Fitting Kmeans model..')
# #Kmeans
# for num in range(2, n):
#     evaluate_kmeans(KMeans(init='k-means++', n_clusters=num, n_init=10),
#                   cluster=num, data=reduced_data,algo='Kmeans_PCA',writer=writer)


# ofile.close()
# print('Kmeans closed')

# ofile = open('minst_em_pca.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'bic','aic','algo'])

# print('Fitting EM model..')
# # EM
# for num in range(2, n):
#     evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
#                   cluster=num, data=train,algo='EM_PCA', writer=writer,test=test)

# print('EM Complete')
# print('PCA Complete')
# ofile.close()


# # ICA
# #-----------------------------------
# print('Fitting ICA model..')
# ofile = open('minst_kmeans_ica.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score', 'calinski-harabaz score','log-likelihood','algo'])

# reduced_data = FastICA(n_components=reduced_to,algorithm='parallel').fit_transform(minst)
# # cluster_viz_km(reduced_data,8,'ICA')

# #KMeans
# for num in range(2, n):
#     evaluate_kmeans(KMeans(init='k-means++', n_clusters=num, n_init=10),
#                   cluster=num, data=reduced_data,algo='Kmeans_ICA',writer=writer)

# print('Completed ICA model')
# ofile.close()

# ofile = open('minst_em_ica.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score','bic','aic','algo'])

# print('Fitting EM model..')
# # EM
# for num in range(2, n):
#     evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
#                   cluster=num, data=train,algo='EM',writer=writer,test=test)

# print('EM Complete')

# ofile.close()

# Randomized Projections
#------------------------------------
# print('Fitting Randomized Projection model..')
# ofile = open('minst_kmeans_randomized_projections.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score', 'calinski-harabaz score','log-likelihood','algo'])
# reduced_data = GaussianRandomProjection(n_components=reduced_to, eps=0.1, random_state=None).fit_transform(minst)
# # cluster_viz_km(reduced_data,8,'Randomized Projection')
# #KMeans
# print ('Kmeans ')
# for num in range(2, n):
#     evaluate_kmeans( KMeans(init='k-means++', n_clusters=num, n_init=10),
#                   cluster=num, data=reduced_data,algo='Kmeans_RP',writer=writer)

# print('Completed Kmeans Randomized Proje model')
# ofile.close()

# ofile = open('minst_em_randomized_projections.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score','bic','aic','algo'])

# print('Fitting EM model..')
# # EM
# for num in range(2, n):
#     evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
#                   cluster=num, data=train,algo='EM',writer=writer,test=test)

# print('EM Complete')

# ofile.close()


# # LDA
#-----------------------------------------
# print('Fitting Factor Analysis model..')
# ofile = open('minst_kmeans_svd.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score', 'calinski-harabaz score','algo'])
# reduced_data = FactorAnalysis(n_components=reduced_to, svd_method='randomized').fit_transform(minst)
# cluster_viz_km(reduced_data,8,'Factor Analysis')
# #KMeans
# for num in range(2, n):
#     evaluate_kmeans( KMeans(init='k-means++', n_clusters=num, n_init=10),
#                   cluster=num, data=reduced_data,algo='KM_FA',writer=writer)

# print('Completed SVD model')
# ofile.close()

# ofile = open('minst_em_svd.csv', "w")
# writer = csv.writer(ofile, delimiter=',')
# writer.writerow(['k', 'time', 'silhouette score','bic','aic','algo'])

# print('Fitting EM model..')
# # EM
# for num in range(2, n):
#     evaluate_em(mixture.GaussianMixture(n_components=num, covariance_type='full'),
#               cluster=num, data=train,algo='EM_FA',writer=writer,test=test)

# print('EM Complete')

# ofile.close()

