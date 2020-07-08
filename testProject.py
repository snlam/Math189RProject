# for timing our program
import timeit
start = timeit.default_timer()

# for natural language processing things (i.e. word2Vec)
import spacy
nlp = spacy.load("en_core_web_md") # sm / md / lg refers to size of the modules; lg [788 MB] vs sm [10 MB] will be accurate, but loads a lot slower

# for mathy operations
import math, numpy

# required to open dataset (gzip) and grab each data point (json) 
import gzip, json



def vec(s):
    """ takes in a word (string), outputs its vector representation """
    # return nlp(s).vector # if loading sm spacy module
    return nlp.vocab[s].vector # if loading md / lg spacy module


def meanv(coords):
    """ takes in a list of N-dimensional vectors, outputs the mean vector 
    (assumes every item in coords has same length as item 0) """
    
    sumv = [0] * len(coords[0])     # gives a 0 vector of dimension = dim(coords[0])
    for item in coords:
        for i in range(len(item)):
            sumv[i] += item[i]
    mean = [0] * len(sumv)          # gives a 0 vector of dimension = dim(sumv) = dim(coords[0])
    for i in range(len(sumv)):
        mean[i] = float(sumv[i]) / len(coords)
    return mean


def dist(a,b):
    """ takes in two N-dimensional vectors: a and b. Outputs the distance from a to b. """
    return numpy.linalg.norm(a-b)



if __name__ == '__main__':
    """ the k-clustering algorithm """
    

    # ============= STEP 0: LOADING DATA ================= #
    # ~30 sec.

    start_step0 = timeit.default_timer()

    all_dataPoints = []
    
    print('==> Loading poetry line data...')
    
    for datapoint in gzip.open("gutenberg-poetry-v001.ndjson.gz"):
        all_dataPoints.append(json.loads(datapoint.strip()))
        
    all_poetryLines = [datapoint['s'] for datapoint in all_dataPoints]  # creates a list of just the poetry lines from the data points

    stop_step0 = timeit.default_timer()
    print('Time Elapsed While Loading Data (STEP 0): ', stop_step0 - start_step0,'sec') 



    # ============= STEP 1: DATA -> VECTORS ================= #
    # ~?? sec.

    start_step1 = timeit.default_timer()

    all_lineVectors = []

    print('\n==> Conerting list of poetry lines to list of vectors...')
    lineTracker = 0 # communicates how many lines have been vectorized
    maxLines = 500000 # 500,000 for the sake of minimum project requirement (otherwise loop takes forever)

    for poetryLine in all_poetryLines[:maxLines]: 

        lineVectorization = []
        doc = nlp(poetryLine)
        
        for token in doc:
            if token.is_alpha:
                wordVec = vec(token.text)
                lineVectorization += [wordVec] # TODO: make sure each wordVec is of same dimension, otherwise meanv function will not work correctly
        
        lineVec = meanv(lineVectorization)
        all_lineVectors.append(lineVec)

        lineTracker += 1
        if lineTracker % 50000 == 0:
            print('\n ~ Only ', maxLines - lineTracker, 'lines left! ~')


    stop_step1 = timeit.default_timer()
    print('Time Elapsed While Converting Poetry Lines to Vectors (STEP 1) : ', stop_step1 - start_step1,'sec')  



    # ============= STEP 2: K-CLUSTERING ================= #
    # ~?? sec.

    start_step2 = timeit.default_timer()

    stop_step2 = timeit.default_timer()
    print('Time Elapsed While K-Clustering (STEP 2) : ', stop_step2 - start_step2,'sec') 

# stops the timer
stop = timeit.default_timer()
print('Total Time Elapsed: ', stop - start,'sec')  