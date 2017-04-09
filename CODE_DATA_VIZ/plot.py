print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

data = pandas.read_csv(
    "/Users/lusenii/Google Drive/Assignment_3/data/minst/KM/minst_kmeans.csv",low_memory=False)

x = data['k']
y = data['silhouette score']
print(y)
plt.bar(x,y)
plt.show()
# n_digits = len(np.unique(digits.target))
# labels = digits.target

