"""Practical 1

Greatly inspired by Stanford CS224 2019 class.
"""

import sys

import pprint

import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import random
import nltk

nltk.download('reuters')
nltk.download('pl196x')
import random

import numpy as np
import scipy as sp
from nltk.corpus import reuters
from nltk.corpus.reader import pl196x
from sklearn.decomposition import PCA, TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


#################################
# TODO: a)
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the 
            corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the 
            corpus
    """
    corpus_words = []
    num_corpus_words = -1
    # ------------------
    # Write your implementation here.
    words_set = set()
    for doc in corpus:
        for word in doc:
            words_set.add(word)
    corpus_words = sorted(list(words_set))
    corpus_words.sort()
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# ---------------------

# Define toy corpus
test_corpus = ["START ala miec kot i pies END".split(" "),
               "START ala lubic kot END".split(" ")]

test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted(list(set([
    'ala', 'END', 'START', 'i', 'kot', 'lubic', 'miec', 'pies'])))
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

#################################
# TODO: b)
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window.
            Words near edges will have a smaller number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the 
                same as the ordering of the words given by the distinct_words 
                function.
            word2Ind (dict): dictionary that maps word to index 
                (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = {}

    # ------------------
    # Write your implementation here.
    for i in range(num_words):
        word2Ind[words[i]] = i

    for doc in corpus:
        for base_ind in range(len(doc)):
            begin = max(base_ind - window_size, 0)
            end = min(base_ind + window_size + 1, len(doc))
            for ind in range(begin, end):
                if base_ind != ind:
                    M[word2Ind[doc[base_ind]]][word2Ind[doc[ind]]] += 1

    # ------------------

    return M, word2Ind

# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# ---------------------

# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["START ala miec kot i pies END".split(" "),
               "START ala lubic kot END".split(" ")]

M_test, word2Ind_test = compute_co_occurrence_matrix(
    test_corpus, window_size=1)

# Correct M and word2Ind
M_test_ans = np.array([
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [0., 0., 2., 0., 0., 0., 0., 0.],
    [0., 2., 0., 0., 0., 1., 1., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [1., 0., 0., 1., 0., 1., 1., 0.],
    [0., 0., 1., 0., 1., 0., 0., 0.],
    [0., 0., 1., 0., 1., 0., 0., 0.],
    [1., 0., 0., 1., 0., 0., 0., 0.]
])

word2Ind_ans = {'END': 0, 'START': 1, 'ala': 2, 'i': 3, 'kot': 4, 'lubic': 5, 'miec': 6, 'pies': 7}

# Test correct word2Ind
assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans, word2Ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2Ind_ans.keys():
    idx1 = word2Ind_ans[w1]
    for w2 in word2Ind_ans.keys():
        idx2 = word2Ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Matrix Tests!")
print ("-" * 80)

#################################
# TODO: c)
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality
        (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following
         SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number 
                of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)):
            matrix of k-dimensioal word embeddings.
            In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    M_reduced = svd.fit_transform(M)
    # ------------------

    print("Done.")
    return M_reduced

# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness 
# In fact we only check that your M_reduced has the right dimensions.
# ---------------------

# Define toy corpus and run student code
test_corpus = ["START ala miec kot i pies END".split(" "),
               "START ala lubic kot END".split(" ")]
M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 8), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 8)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

#################################
# TODO: d)
def plot_embeddings(M_reduced, word2Ind, words, save = False, filename = ''):
    """ Plot in a scatterplot the embeddings of the words specified 
        in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the
            corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to
            visualize
    """

    # ------------------
    # Write your implementation here.
    xs = []
    ys = []

    for word in words:
        [x, y] = M_reduced[word2Ind[word]]
        xs.append(x)
        ys.append(y)

    fig, ax = plt.subplots()
    ax.scatter(xs, ys)

    for word in words:
        [x, y] = M_reduced[word2Ind[word]]
        ax.annotate(word, (x, y))

    if save:
        plt.savefig(filename+'png')
    else:
        plt.show()
    # ------------------#

# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# The plot produced should look like the "test solution plot" depicted below. 
# ---------------------

print ("-" * 80)
print ("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2Ind_plot_test = {
    'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words)

print ("-" * 80)

#################################
# TODO: e)
# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------

def read_corpus_pl():
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    pl196x_dir = nltk.data.find('corpora/pl196x')
    pl = pl196x.Pl196xCorpusReader(
        pl196x_dir, r'.*\.xml', textids='textids.txt',cat_file="cats.txt")
    tsents = pl.tagged_sents(fileids=pl.fileids(),categories='cats.txt')[:5000]

    return [[START_TOKEN] + [
        w[0].lower() for w in list(sent)] + [END_TOKEN] for sent in tsents]


def plot_unnormalized(corpus, words):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    plot_embeddings(M_reduced_co_occurrence, word2Ind_co_occurrence, words, True, 'plot_unnormalized')


def plot_normalized(corpus, words):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, True, 'plot_normalized')
'''
pl_corpus = read_corpus_pl()
words = [
    "sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]

plot_normalized(pl_corpus, words) #TODO: describe plot
plot_unnormalized(pl_corpus, words) #TODO: describe plot

'''
#################################
# Section 2:
#################################
# Then run the following to load the word2vec vectors into memory. 
# Note: This might take several minutes.
wv_from_bin_pl = KeyedVectors.load("word2vec_100_3_polish.bin")

# -----------------------------------
# Run Cell to Load Word Vectors
# Note: This may take several minutes
# -----------------------------------


#################################
# TODO: a)
def get_matrix_of_vectors(wv_from_bin, required_words):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors
                         loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    words = list(wv_from_bin.key_to_index.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind

# -----------------------------------------------------------------
# Run Cell to Reduce 300-Dimensinal Word Embeddings to k Dimensions
# Note: This may take several minutes
# -----------------------------------------------------------------

#################################
# TODO: a)
words = [
    "sztuka", "śpiewaczka", "literatura", "poeta", "artystyczny", "obywatel"]

M_wv_pl, word2Ind_wv_pl = get_matrix_of_vectors(wv_from_bin_pl, words)
M_reduced = reduce_to_k_dim(M_wv_pl, k=2)


plot_embeddings(M_reduced, word2Ind_wv_pl, words, True, 'word2vec_plot')


#################################
# TODO: b)
# Polysemous Words
# ------------------
# Write your polysemous word exploration code here.

def polysemeous_exploration(word):
    polysemous = wv_from_bin_pl.most_similar(word)
    print("Polysemeus word exploration - words similar to: " + word)
    for i in range(10):
        key, similarity = polysemous[i]
        print(i, key, similarity)

polysemeous_exploration("stówa")
#polysemeous_exploration("myszka") # expected computer and animal, got only animal
#polysemeous_exploration("pączek") expected desserts, got fruits and plant parts
#polysemeous_exploration("zamek") expected results for castle and zip, got only castle
#polysemeous_exploration("para") expected steam and couple/two, got only couple /two
#polysemeous_exploration("pokój") expected room and peace, got room
polysemeous_exploration("blok") # sucescc! - got both building-related and cube related words
#polysemeous_exploration("język") # hoped for tounge and language, got only language (and little hedgehog)
#polysemeous_exploration("pilot") only pilot, not remote
#polysemeous_exploration("kanar") hoped for bird and ticket checking person, somehow got neither
# ------------------

#################################
# TODO: c)
# Synonyms & Antonyms
# ------------------
# Write your synonym & antonym exploration code here.
def synonym_antonym_exploration(w1, w2, w3):
    w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
    w1_w3_dist = wv_from_bin_pl.distance(w1, w3)

    print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
    print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))
w1 = "radosny"
w2 = "pogodny"
w3 = "smutny"

w1 = "mały"
w2 = "drobny"
w3 = "duży"

synonym_antonym_exploration("radosny", "pogodny", "smutny")

synonym_antonym_exploration("mały", "drobny", "duży")

synonym_antonym_exploration("chudy", "wysmukły", "gruby")

synonym_antonym_exploration("jasny", "świetlisty", "ciemny")
# why - the antonyms I used are the most 'basic' words describing a certain quality (for exaple size)
# the synonym I used is a word that describes that quality but is less commonly used and conveys some additional meanings:
quit()
#################################
# TODO: d)
# Solving Analogies with Word Vectors
# ------------------

# ------------------
# Write your analogy exploration code here.
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["syn", "kobieta"], negative=["mezczyzna"]))


#################################
# TODO: e)
# Incorrect Analogy
# ------------------
# Write your incorrect analogy exploration code here.

# ------------------


#################################
# TODO: f)
# Guided Analysis of Bias in Word Vectors
# Here `positive` indicates the list of words to be similar to and 
# `negative` indicates the list of words to be most dissimilar from.
# ------------------
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'szef'], negative=['mezczyzna']))
print()
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['mezczyzna', 'prezes'], negative=['kobieta']))


#################################
# TODO: g)
# Independent Analysis of Bias in Word Vectors 
# ------------------


#################################
# Section 3:
# English part
#################################
wv_from_bin = load_word2vec()

#################################
# TODO: 
# Find English equivalent examples for points b) to g).
#################################
