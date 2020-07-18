# for timing our program
import timeit

# for natural language processing things in K-Means algorithm
import spacy
nlp = spacy.load("en_core_web_lg") # sm / md / lg refers to size of the modules; lg [788 MB] vs sm [10 MB] will be accurate, but loads a lot slower

# for sentiment analysis
import nltk
from textblob import TextBlob

# for mathy operations
import math 
import numpy as np

# for measuring cosine distance
from scipy.spatial import distance

# required to open dataset (gzip) and grab each data point (json) 
import gzip, json

# required to use 'deepcopy' command; taken from https://mubaris.com/posts/kmeans-clustering/
from copy import deepcopy

# used to write and read .csv files
import csv

# for making plots
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS
wordDim = 300  # dimension of word vectors


def vec(s):
    """ takes in a word (string), outputs its vector representation """
    
    # if loading sm spacy module
    # wordVec = nlp(s).vector 

    # lg s if loading md /pacy module
    wordVec = nlp.vocab[s].vector 

    return wordVec


def meanv(coords):
    """ takes in a list of N-dimensional vectors, outputs the mean vector 
    (assumes every item in coords has same length as item 0) """

    # summing all vectors component-wise
    sumv = [0] * wordDim 
    for item in coords:
        for i in range(len(item)):
            sumv[i] += item[i]

    # dividing by len(coords) component-wise to get the mean vector
    mean = [0] * wordDim
    for i in range(len(sumv)):
        mean[i] = float(sumv[i]) / len(coords)

    return mean


def dist(a, b, ax = 1):
    """ takes in two N-dimensional vectors: a and b. Outputs the distance from a to b. """

    return np.linalg.norm(a-b, axis = ax)


def maxNorm(vectors):
    """ takes in a list of vectors; outputs tuple of 1) vector with largest norm and 2) its norm """

    maxNorm = 0
    maxVector = [0]*300

    # caculates norm of each vector is 'vectors'; saves the info of largest one
    for vector in vectors:
        norm = np.linalg.norm(vector)
        if norm > maxNorm:
            maxNorm = norm
            maxVector = vector

    return maxVector, maxNorm


def getSpaCyWords():
    """ lists all word vectors in SpaCy vocabulary;
    output: a list of all word vectors in SpaCy vocabulary and their id-s """
    
    # Format the vocabulary for use in the distance function
    ids = [x for x in nlp.vocab.vectors.keys()]
    vocabulary = [nlp.vocab.vectors[x] for x in ids]
    vocabulary = np.array(vocabulary)

    return vocabulary, ids


def closestWord(A_2d, vocabulary, ids):
    """ finds the closest word to a 2 dimensional array of 300 coordinates;
    input: 2D array of 300 coordinates, list of vectorized vocabulary; output: word closest (cosine) to input array """

    # measuring which word is in closeset distance to ith centroid in terms of cosine similarity
    closest_index = (1 - distance.cdist(A_2d, vocabulary, metric='cosine')).argmax()
    word_id = ids[closest_index]
    output_word = nlp.vocab[word_id].text

    return output_word


def dataLoader():
    """ loads the Gutenberg Poetry dataset;
    output: a list of all poetry lines loaded """

    # TODO: figure out how to limit the number of lines LOADED, instead of limiting the number of lines TOKENIZED

    print('\n==> Loading poetry line data...')

    all_dataPoints = []
    for datapoint in gzip.open("gutenberg-poetry-v001.ndjson.gz"):
        all_dataPoints.append(json.loads(datapoint.strip()))
    
    # creates a list of the poetry lines as strings from the data
    all_poetryLines = [datapoint['s'] for datapoint in all_dataPoints]

    return all_poetryLines


def dataTokenizer(all_poetryLines, maxLines = 1000):
    """ loads and tokenizes the Project Gutenberg poetry corpus (found here: https://github.com/aparrish/gutenberg-poetry-corpus); 
    input: number of lines to load from the dataset; output: list of vectors corresponding to each poetry line (currently using mean vector) """

    # TOKENIZATION PROCESS:
    # ~1 hr 45 min...(50,000 lines)
    # ~35 sec...(5000 lines)

    print('==> Converting poetry lines to vectors...')

    all_lineVectors = []
    for poetryLine in all_poetryLines[:maxLines]: 

        lineVectorization = []
        doc = nlp(poetryLine)
        
        # tokenization process 
        # deleting only stop words and non-alphabetical tokens from poetry lines
        for token in doc:
            if (token.is_alpha & (not token.is_stop)): 
                wordVec = vec(token.text)
                lineVectorization += [wordVec]
        
        # discard poetry line if tokens were only stop words, non-alphabetical tokens, or blank spaces
        # TODO: it would be interesting to know how many datapoints we are discarding in this way.
        if len(lineVectorization) == 0:
            continue 

        # calculating vector representation of poetry line and adding to list
        lineVec = meanv(lineVectorization)
        all_lineVectors.append(lineVec)
    
    return all_lineVectors


def kMeansAlg(K, vocabulary, ids, all_lineVectors):
    """ the k-clustering algorithm; input: number of centroids to initialize (K), list of vector data (all_lineVectors)
    list of id-s of all word vectors in SpaCy library (ids), list of vectorized SpaCy vocabulary (vocabulary) """

    start_alg = timeit.default_timer()

    # ============= K-CLUSTERING ================= #
    # ~7 min...(50,000 lines)

    print('==> Commencing K-clustering algorithm...')

    
    all_lineVectors_array = np.array(all_lineVectors) 

    # storing vector with largest norm and its norm
    largestVector, largestNorm = maxNorm(all_lineVectors)

    # initializing array of I.D.-s tying each vector in dataset to its cluster
    listOfClusterLabels = np.zeros(len(all_lineVectors_array))


    Centroids = []
    Centroids_old = []
    for i in range(K):

        # initializing centroids with random coordinates
        ith_centroid = np.random.rand(wordDim) * max(largestVector)  # TODO: is this the best way to randomly generate numbers (i.e. is uniform best)?

        Centroids.append(ith_centroid)
        Centroids_old.append(np.zeros(wordDim))

    # these are now 2 dimensional arrays; Centroids[i] corresponds to the array of the ith centroid's coordinates (same for _old)
    Centroids = np.array(Centroids)
    Centroids_old = np.array(Centroids_old)

    # comparison function of K-Means: calculates distance between updated centroids and old centroids
    error = dist(Centroids, Centroids_old, None)

    # the convergence algorithm (loop until centroids and old centroids are equivalent)
    loopTracker = 0
    while error != 0:
        
        # Assigning each value to its closest cluster
        for i in range(len(all_lineVectors_array)):
            
            # array where each member tells us how faraway the ith data point is from each centroid
            distances = dist(all_lineVectors_array[i], Centroids)
            
            # update list which tells us which cluster (0, ..., K) each data point belongs to
            clusterLabel = np.argmin(distances)
            listOfClusterLabels[i] = clusterLabel

        # should always store the initialized Centroids_old array -- used to start algorithm over if a cluster has 0 datapoints
        Centroids_orig = deepcopy(Centroids_old)

        # Storing previous centroid array
        Centroids_old = deepcopy(Centroids)
      
        # Finding the new centroids by taking the average value of the datapoints belonging to it
        arrayof_NumberOfPointsInEachCluster = np.zeros(K)
        listof_PointsInEachCluster = []
        for i in range(K):

            # creates a list of all vectors in the ith cluster
            points_ithCluster = [all_lineVectors_array[j] for j in range(len(all_lineVectors_array)) if listOfClusterLabels[j] == i]

            # if a cluster has 0 datapoints, start the algorithm over
            if len(points_ithCluster) == 0:
                for j in range(K):
                    Centroids[j] = np.random.rand(wordDim) * max(largestVector)
                    Centroids_old = deepcopy(Centroids_orig)
                break

            arrayof_NumberOfPointsInEachCluster[i] = len(points_ithCluster)
            listof_PointsInEachCluster.append(points_ithCluster)

            # updating ith centroid
            Centroids[i] = np.mean(points_ithCluster, axis=0)

        # calculating distance from previous centroids
        error = dist(Centroids, Centroids_old, None)
    
        # debugging for infinite loops
        loopTracker += 1
        if loopTracker > 1000:
            print(' ** ** ** !! RUH ROH !! ** ** **')
            break


    # ============= OUTPUTS ================= #

    print('==> Calculating algorithm outputs...')

    totalVariance = 0
    listOfOutputs = []

    for i in range(K):

        # take the points belonging to the ith finalized cluster
        points_ithCluster = listof_PointsInEachCluster[i]

        # convert the list of vectors into an array (becomes a 2D array for closestword() function)
        points_ithCluster = np.array(points_ithCluster)
        
        # debugging
        if len(points_ithCluster) == 0:
            print("Something didn't work...")

        else:

            # measuring which word is in closest distance to ith centroid in terms of cosine similarity
            # we use np.array(Centroids[i]) because Centroids[i] is a 1 dimensional array of 300 features; closestWord func requires 2D array 
            output_word = closestWord( np.array([Centroids[i]]) , vocabulary, ids)

            # calculating variance
            var_ithCluster = np.var(points_ithCluster)
            totalVariance += var_ithCluster
        
        # algorithm outputs: norm of centroid, number of points in its cluster, the output word
        listOfOutputs.append((np.linalg.norm(Centroids,axis=1)[i], len(points_ithCluster), output_word))

            
    # stops the timer
    stop_alg = timeit.default_timer()
    time = stop_alg - start_alg


    return totalVariance, listOfOutputs, time



if __name__ == '__main__':

    start = timeit.default_timer()

    # SET PARAMETERS OF RUN
    lines = 500000   # number of poetry lines being tested (out of 3,000,000)...
    largestK = 10   # testing all K <= largestK...
    N = 15           # then for each such K value, perform THIS many clustering attempts


    # ============= STEP 1: LOADING DATASET ================= #

    # formatting the SpaCy vocabulary; a list of all its word vectors 
    vocabulary, ids = getSpaCyWords()

    # loading the poetry dataset; returns a list of strings (the poetry lines)
    all_poetryLines = dataLoader()


    # ============= STEP 2: TOKENIZATION ================= #

    # tokenizing the usable poetry lines
    all_lineVectors = dataTokenizer(all_poetryLines, maxLines = lines)


    # ============= STEP 3: K-MEANS ALGORITHM ================= #

    # array of best total variances from each clustering attempt
    bestVariances = np.zeros(largestK)

    # list collecting the output tuples (norm of centroid, number of points in its cluster, and the output word) from the best clustering attempts
    infoFrom_bestAttemptsPerK = []

    # iterating the K-Means algorithm over every 1 <= K <= largestK
    for K in range(1,largestK+1):

        # initializing lists to collect the outputs (total variance, listOfOutputs, and runtime) from each K-Means algorithm run
        totalVarianceList = []
        listOf_listOfOutputs = []
        listOfTimes = []

        # for every K, run the algorithm N many times
        for i in range(N):

            print('\n  K =',K,'; N =',i+1,'of', N,'\n')

            # !! CALLING THE ALGORITHM HERE.
            totalVariance, listOfOutputs, time = kMeansAlg(K, vocabulary, ids, all_lineVectors)

            # add the called info into their respective lists
            totalVarianceList += [totalVariance]
            listOf_listOfOutputs += [listOfOutputs]
            listOfTimes += [time]


        # !! defining the "best clustering attempts" as the clusterings that minimized total variance !!


        # record the variance from the run with the minimum variance
        bestVariances[K-1] = min(totalVarianceList)
        
        # record the list of outputs from the best clustering attempt
        index_minVar = totalVarianceList.index( min(totalVarianceList) )
        infoFrom_bestAttemptsPerK.append( listOf_listOfOutputs[index_minVar] )

        # print statements for debugging (from all N runs)

        print('\n------------------------------------------------------------')

        print('Showing Record of', N, 'Clustering Attempts at K =',K,'...\n')
        for i in range(N):
            print('Attempt', i+1,':')
            print('    Total Variance:', totalVarianceList[i]) 
            print('    List of Outputs:')
            for j in range(K):
                print('        ',j+1,':', listOf_listOfOutputs[i][j])
            print('    Time Elapsed:', listOfTimes[i],'sec\n')

        print('------------------------------------------------------------')


    # ============= STEP 4: PRINTING BEST CLUSTERING ATTEMPTS ================= #

    print('\n')
    
    for K in range(1,largestK+1):

        # the important stuff (the best clustering attempt for the K value)
        print('K =', K, 'BEST CLUSTERING ATTEMPT:  \n')
        
        minVar_listOfOutputs = infoFrom_bestAttemptsPerK[K-1]

        for i in range(K):
            info_ithCentroid = minVar_listOfOutputs[i]
            print('     Norm of Centroid',i+1,':', info_ithCentroid[0])
            print('     Clustered', info_ithCentroid[1], 'lines of poetry.')
            print('          *-->', info_ithCentroid[2],'<--*\n')


    # ============= STEP 5: MAKING PLOTS ================= #

    # 1) Reduction in Var. vs. K

    # initializing x-axis (K vals)
    kvalues = np.arange(1, largestK+1, 1)

    # initializing y-axis (red. in total variance)
    reduction_inVariation = np.zeros(largestK)
    for K in range(1,largestK):
        reduction_inVariation[K] = bestVariances[K-1] - bestVariances[K]

    # plotting red. in total variance vs. K
    plt.plot(kvalues, reduction_inVariation, 'r-')
    plt.grid(axis='y', alpha=0.5)
    plt.ylabel('Reduction in Variance')
    plt.xlabel('Number of Clusters (K)')
    plt.title('Reduction in Variance vs. Number of Clusters')
    plt.text( max(kvalues) * 3/4, min(reduction_inVariation) * 1/4, "%i lines\n$N = %i$" % (lines, N), bbox=dict(facecolor='red', alpha=0.3), fontsize = 18)
    plt.xticks(np.arange(min(kvalues), max(kvalues)+1, 1.0))
    plt.show()

    # 2) Sentiment Analysis

    # initialize lists for histogram creation
    allPolarity_vals = []
    allSubjectivity_vals = []

    for line in all_poetryLines:
        
        # returns a 2-tuple of polarity value and subjectivity value
        lineSentiment = TextBlob(line).sentiment

        # record the polarity value of the line (between -1.0 to 1.0)
        linePolarity = lineSentiment[0]
        allPolarity_vals.append(linePolarity)

        # record the subjectivity value of the line (between 0.0 and 1.0)
        lineSubjectivity = lineSentiment[1]
        allSubjectivity_vals.append(lineSubjectivity)

    # average polarity and subjectivity scores of all poetry lines
    meanPolarity = sum(allPolarity_vals) / len(all_poetryLines)
    meanSubjectivity = sum(allSubjectivity_vals) / len(all_poetryLines)

    # polarity histogram
    plt.hist(allPolarity_vals, bins = 50)
    plt.grid(axis='y', alpha=0.75)
    plt.yscale('log', nonposy = 'clip')
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Polarity Scores')
    plt.text(.5, 1500000, r'$\mu = %i$' %meanPolarity, bbox=dict(facecolor='red', alpha=0.3), fontsize = 18)
    plt.show()

    # subjectivity histogram
    plt.hist(allSubjectivity_vals, bins = 50)
    plt.grid(axis='y', alpha=0.75)
    plt.yscale('log', nonposy = 'clip')
    plt.xlabel('Subjectivity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Subjectivity Scores')
    plt.text(.75, 1350000, r'$\mu = %i$' %meanSubjectivity, bbox=dict(facecolor='red', alpha=0.3), fontsize = 18)
    plt.show()

    print("Average Polarity of Poetry Lines in Dataset:", meanPolarity )
    print("Average Subjectivity of Poetry Lines in Dataset:", meanSubjectivity )

    # end time record
    stop = timeit.default_timer()
    print('Total Time Elapsed: ', stop - start,'sec') 

    # TIME LOGS:
    # 200 lines; K = 1; N = 1; ~
    # 1000 lines; K = 2; N = 2: ~55sec
    # SA: ~22min
