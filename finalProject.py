# for timing our program
import timeit
start = timeit.default_timer()

# for natural language processing things (i.e. word2Vec)
import spacy
nlp = spacy.load("en_core_web_lg") # sm / md / lg refers to size of the modules; lg [788 MB] vs sm [10 MB] will be accurate, but loads a lot slower
wordDim = 300    # all vector representations of words in spacy vocabulary are of dimension 300

# for mathy operations
import math 
import numpy as np

# required to open dataset (gzip) and grab each data point (json) 
import gzip, json

# required to use 'deepcopy' command; taken from https://mubaris.com/posts/kmeans-clustering/
from copy import deepcopy



def vec(s):
    """ takes in a word (string), outputs its vector representation """
    # return nlp(s).vector # if loading sm spacy module
    return nlp.vocab[s].vector # if loading md / lg spacy module


def meanv(coords):
    """ takes in a list of N-dimensional vectors, outputs the mean vector 
    (assumes every item in coords has same length as item 0) """
    
    # sumv = [0] * len(coords[0])     # gives a 0 vector of dimension = dim(coords[0])
    sumv = [0] * wordDim              # 300 dimensional word vectors
    for item in coords:
        for i in range(len(item)):
            sumv[i] += item[i]
    # mean = [0] * len(sumv)          # gives a 0 vector of dimension = dim(sumv) = dim(coords[0])
    mean = [0] * wordDim
    for i in range(len(sumv)):
        mean[i] = float(sumv[i]) / len(coords)
    return mean


def dist(a, b, ax = 1):
    """ takes in two N-dimensional vectors: a and b. Outputs the distance from a to b. """
    return np.linalg.norm(a-b, axis = ax)


def maxNorm(vectors):
    """ takes in a list of vectors; outputs tuple of vector with largest norm and its respective norm """
    maxNorm = 0
    maxVector = [0]*300
    for vector in vectors:
        norm = np.linalg.norm(vector)
        if norm > maxNorm:
            maxNorm = norm
            maxVector = vector

    return maxVector, maxNorm


if __name__ == '__main__':
    """ the k-clustering algorithm """
    
    # ============= STEP 0: LOADING DATA ================= #
    # ~30 - 45 sec.

    start_step0 = timeit.default_timer()

    all_dataPoints = []
    
    print('\n==> Loading poetry line data...')
    
    for datapoint in gzip.open("gutenberg-poetry-v001.ndjson.gz"):
        all_dataPoints.append(json.loads(datapoint.strip()))
        
    all_poetryLines = [datapoint['s'] for datapoint in all_dataPoints]  # creates a list of just the poetry lines from the data points

    stop_step0 = timeit.default_timer()
    print('Time Elapsed While Loading Data (STEP 0): ', stop_step0 - start_step0,'sec') 


    # ============= STEP 1: DATA -> VECTORS ================= #
    # ~2 hrs...(for the 50,000 minimum)
    # ~1 - 1.5 min...(for 5000 lines)

    start_step1 = timeit.default_timer()

    all_lineVectors = []

    print('\n==> Converting list of poetry lines to list of vectors...')
    lineTracker = 0 # communicates how many lines have been vectorized
    maxLines = 5000 # 500,000 for the sake of minimum project requirement (otherwise loop takes forever)
    checker = 1000 # have the comp holler back every 'checker' amount of lines tokenized

    for poetryLine in all_poetryLines[:maxLines]: 

        lineVectorization = []
        doc = nlp(poetryLine)
        
        for token in doc:
            if token.is_alpha:
                wordVec = vec(token.text)
                lineVectorization += [wordVec]
        
        if len(lineVectorization) == 0:
            continue

        lineVec = meanv(lineVectorization)
        all_lineVectors.append(lineVec)

        lineTracker += 1
        if lineTracker % checker == 0:
            print('\n ~ Only', maxLines - lineTracker, 'lines left! ~')

    all_lineVectors_array = np.array(all_lineVectors)

    stop_step1 = timeit.default_timer()
    print('Time Elapsed While Converting Poetry Lines to Vectors (STEP 1) : ', stop_step1 - start_step1,'sec')  


    # ============= STEP 2: K-CLUSTERING ================= #
    # ~2 min. (@K = 3; 'md' spacy module; 5000 poetry lines)
    # ~2 min. (@K = 6; 'md' spacy module; 5000 poetry lines)

    start_step2 = timeit.default_timer()

    print('\n==> Commencing K-clustering algorithm...')

    # calculating vector with largest norm and its norm
    largestVector, largestNorm = maxNorm(all_lineVectors)
        
    # number of clusters
    K = 6

    # initializing the "list" (array) of all the labels (clusters) corresponding to each vector (poetry lines)
    listOfClusterLabels = np.zeros(len(all_lineVectors_array))

    # creating list of K randomly generated centroids
    i = 0
    listOfCentroids = []
    listOfCentroids_old = []
    for i in range(K):
        C_i = np.random.randint(0, largestNorm, size = wordDim)
        listOfCentroids.append(C_i)
        listOfCentroids_old.append(np.zeros(wordDim))

    Centroids = np.array(listOfCentroids)
    Centroids_old = np.array(listOfCentroids_old)


    # error func. - distance between new centroids and old centroids
    error = dist(Centroids, Centroids_old, None)


    loopTracker = 0
    # loop will run till the error becomes zero
    while error != 0:

        # Assigning each value to its closest cluster
        for i in range(len(all_lineVectors_array)):
            
            # gives us an array where each member tells us how faraway the data point is from each centroid
            distances = dist(all_lineVectors_array[i], Centroids) 
            
            # this tells us which cluster (0, ..., K) the vector (aka: the poetry line) corresponds to
            clusterLabel = np.argmin(distances)

            # adding the label above to the list of all labels
            listOfClusterLabels[i] = clusterLabel

        # Storing the old centroid values
        Centroids_old = deepcopy(Centroids)

        # Finding the new centroids by taking the average value
        for i in range(K):
            points_ithCluster = [all_lineVectors_array[j] for j in range(len(all_lineVectors_array)) if listOfClusterLabels[j] == i]
            Centroids[i] = np.mean(points_ithCluster, axis=0)
        
        error = dist(Centroids, Centroids_old, None)
        loopTracker += 1
        print('\n ~ ', loopTracker ,': Converging to better centroid values! ~')

    print(Centroids)

    stop_step2 = timeit.default_timer()
    print('\nTime Elapsed While K-Clustering (STEP 2) : ', stop_step2 - start_step2,'sec') 



# stops the timer
stop = timeit.default_timer()
print('\nTotal Time Elapsed: ', stop - start,'sec')  
