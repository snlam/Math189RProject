# import pandas, matplotlib

import gensim, numpy
import gzip, json
import random
import math

# for timing our program
import timeit

# who knows if we're using this; but goes with the list comprehension step
import re

# for natural language processing things
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")


def vec(s):
    return nlp.vocab[s].vector

def meanv(coords):
    # assumes every item in coords has same length as item 0
    sumv = [0] * len(coords[0])
    for item in coords:
        for i in range(len(item)):
            sumv[i] += item[i]
    mean = [0] * len(sumv)
    for i in range(len(sumv)):
        mean[i] = float(sumv[i]) / len(coords)
    return mean

def dist(a,b):
     return numpy.linalg.norm(a-b)


# starts the timer
start = timeit.default_timer()

all_dataPoints = []
i = 0
while i < 500000:
     for datapoint in gzip.open("gutenberg-poetry-v001.ndjson.gz"):
          all_dataPoints.append(json.loads(datapoint.strip()))
          i += 1

all_lines = [datapoint['s'] for datapoint in all_dataPoints]     


all_tokenizations = []
for line in all_lines:
     doc = nlp(line)
     tokenizedLine = []
     for token in doc:
          if token.is_alpha:
               tokenizedLine += [token.text]
     all_tokenizations.append(tokenizedLine)

print(all_tokenizations)     


all_vectorizations = []
for tokenList in all_tokenizations:
     listOfTokenVectors = []
     for token in tokenList:
          tokenVec = vec(token)
          listOfTokenVectors += [tokenVec]
     all_vectorizations.append(listOfTokenVectors)

meanVectorList = []
for vectorList in all_vectorizations:
     sentenceVector = meanv(vectorList)
     meanVectorList.append(sentenceVector)




# K-MEANS CLUSTERING
%matplotlib inline
from copy import deepcopy
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)


## don't need to plot!

# TODO: remember to take out all the plotting things!

# # Getting the values and plotting it
# f1 = data['V1'].values
# f2 = data['V2'].values
# X = np.array(list(zip(f1, f2)))
# plt.scatter(f1, f2, c='black', s=7)

# # Plotting along with the Centroids
# plt.scatter(f1, f2, c='#050505', s=7)
# plt.scatter(C_x, C_y, marker='*', s=200, c='g')

# TODO: look through this K-clustering stuff (the real deal)

# # To store the value of centroids when it updates
# C_old = np.zeros(C.shape)
# # Cluster Lables(0, 1, 2)
# clusters = np.zeros(len(X))
# # Error func. - Distance between new centroids and old centroids
# error = dist(C, C_old, None)
# # Loop will run till the error becomes zero
# while error != 0:
#     # Assigning each value to its closest cluster
#     for i in range(len(X)):
#         distances = dist(X[i], C)
#         cluster = np.argmin(distances)
#         clusters[i] = cluster
#     # Storing the old centroid values
#     C_old = deepcopy(C)
#     # Finding the new centroids by taking the average value
#     for i in range(k):
#         points = [X[j] for j in range(len(X)) if clusters[j] == i]
#         C[i] = np.mean(points, axis=0)
#     error = dist(C, C_old, None)

# TODO: look through this stuff too. (from example 1 in https://mubaris.com/posts/kmeans-clustering/)

# from sklearn.cluster import KMeans

# # Number of clusters
# kmeans = KMeans(n_clusters=3)
# # Fitting the input data
# kmeans = kmeans.fit(X)
# # Getting the cluster labels
# labels = kmeans.predict(X)
# # Centroid values
# centroids = kmeans.cluster_centers_

# # Comparing with scikit-learn centroids
# print(C) # From Scratch
# print(centroids) # From sci-kit learn

# TODO: this underneath is all plotting too

# colors = ['r', 'g', 'b', 'y', 'c', 'm']
# fig, ax = plt.subplots()
# for i in range(k):
#         points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
#         ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
# ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')




# stops the timer
stop = timeit.default_timer()
print('Time: ', stop - start)  





