import numpy
# import scipy
import matplotlib.pyplot as plt
import pandas as pd
# import gensim
import random

from collections import Counter

import pronouncing
import markovify

import gzip, json
all_lines = []
for line in gzip.open("gutenberg-poetry-v001.ndjson.gz"):
    all_lines.append(json.loads(line.strip()))

import re
flower_lines = [line['s'] for line in all_lines if re.search(r'\bflower\b', line['s'], re.I)]

random.sample(flower_lines, 8)

longest = max([len(x) for x in flower_lines]) # find the length of the longest line
center = longest - len("flower") # and use it to create a "center" offset that will work for all lines

sorted_flower_lines = sorted(
    [line for line in flower_lines if re.search(r"\bflower\b\s\w", line)], # only lines with word following
    key=lambda line: line[re.search(r"\bflower\b\s", line).end():]) # sort on the substring following the match

for line in sorted_flower_lines[350:400]: # change these numbers to see a different slice
    offset = center - re.search(r'\bflower\b', line, re.I).start()
    print((" "*offset)+line) # left-pad the string with spaces to align on "flower"

found_adj = []
for line in flower_lines:
    matches = re.findall(r"(the|a)\s(\b\w+\b)\s(\bflower\b)", line, re.I)
    for match in matches: 
        found_adj.append(match[1])

random.sample(found_adj, 12)

Counter(found_adj).most_common(12)

## Rhymes
source_word = "flowering"
source_word_rhymes = pronouncing.rhymes(source_word)
source_word_rhymes

for line in all_lines:
    text = line['s']
    match = re.search(r'(\b\w+\b)\W*$', text)
    if match:
        last_word = match.group()
        if last_word in source_word_rhymes:
            print(text)

phones = pronouncing.phones_for_word(source_word)[0] # words may have multiple pronunciations, so this returns a list
phones

pronouncing.rhyming_part(phones)

from collections import defaultdict
by_rhyming_part = defaultdict(lambda: defaultdict(list))
for line in all_lines:
    text = line['s']
    if not(32 < len(text) < 48): # only use lines of uniform lengths
        continue
    match = re.search(r'(\b\w+\b)\W*$', text)
    if match:
        last_word = match.group()
        pronunciations = pronouncing.phones_for_word(last_word)
        if len(pronunciations) > 0:
            rhyming_part = pronouncing.rhyming_part(pronunciations[0])
            # group by rhyming phones (for rhymes) and words (to avoid duplicate words)
            by_rhyming_part[rhyming_part][last_word.lower()].append(text)

random_rhyming_part = random.choice(list(by_rhyming_part.keys()))
random_rhyming_part, by_rhyming_part[random_rhyming_part]

rhyme_groups = [group for group in by_rhyming_part.values() if len(group) >= 2]

for i in range(7):
    group = random.choice(rhyme_groups)
    words = random.sample(list(group.keys()), 2)
    print(random.choice(group[words[0]]))
    print(random.choice(group[words[1]]))

## Markov
big_poem = "\n".join([line['s'] for line in random.sample(all_lines, 250000)])

model = markovify.NewlineText(big_poem)

for i in range(14):
    print(model.make_sentence())

model.make_short_sentence(60)

for i in range(6):
    print()
    for i in range(random.randrange(1, 5)):
        print(model.make_short_sentence(40))
    # ensure last line has a period at the end, for closure
    print(re.sub(r"(\w)[^\w.]?$", r"\1.", model.make_short_sentence(40)))
    print()
    print("～ ❀ ～")

# K-MEANS CLUSTERING
# %matplotlib inline
# from copy import deepcopy
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# plt.rcParams['figure.figsize'] = (16, 9)
# plt.style.use('ggplot')

# # Importing the dataset
# data = pd.read_csv('xclara.csv')
# print(data.shape)
# data.head()

# # Getting the values and plotting it
# f1 = data['V1'].values
# f2 = data['V2'].values
# X = np.array(list(zip(f1, f2)))
# plt.scatter(f1, f2, c='black', s=7)

# # Euclidean Distance Caculator
# def dist(a, b, ax=1):
#     return np.linalg.norm(a - b, axis=ax)

# # Number of clusters
# k = 3
# # X coordinates of random centroids
# C_x = np.random.randint(0, np.max(X)-20, size=k)
# # Y coordinates of random centroids
# C_y = np.random.randint(0, np.max(X)-20, size=k)
# C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
# print(C)

# # Plotting along with the Centroids
# plt.scatter(f1, f2, c='#050505', s=7)
# plt.scatter(C_x, C_y, marker='*', s=200, c='g')

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

# colors = ['r', 'g', 'b', 'y', 'c', 'm']
# fig, ax = plt.subplots()
# for i in range(k):
#         points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
#         ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
# ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

