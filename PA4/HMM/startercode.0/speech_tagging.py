import numpy as np
import time
import random
from hmm import HMM

"""
Function to calculate accuracy: correct/total
"""
def accuracy(predict_tagging, true_tagging):
    if len(predict_tagging) != len(true_tagging):
        return 0, 0, 0
    cnt = 0
    for i in range(len(predict_tagging)):
        if predict_tagging[i] == true_tagging[i]:
            cnt += 1
    total_correct = cnt
    total_words = len(predict_tagging)
    if total_words == 0:
        return 0, 0, 0
    return total_correct, total_words, total_correct * 1.0 / total_words

"""
This class is initialized by providing it with tag file, data file, split percentage and random seed
"""
class Dataset:
    def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
        # read data and tags
        tags = self.read_tags(tagfile)
        data = self.read_data(datafile)

        # initialize tag content
        self.tags = tags

        # initialize lines
        lines = []
        for l in data:
            # create a class Line for each new line
            new_line = self.Line(l)

            # if sentence is not empty
            if new_line.length > 0:
                lines.append(new_line)

        # initialize seed if seed is non-existent
        if seed is not None:
            random.seed(seed)

        # split the dataset to test and train data
        random.shuffle(lines)
        train_size = int(train_test_split * len(data))
        self.train_data = lines[:train_size]
        self.test_data = lines[train_size:]
        return

    def read_data(self, filename):
        """Read tagged sentence data"""
        with open(filename, 'r') as f:
            sentence_lines = f.read().split("\n\n")
        return sentence_lines

    def read_tags(self, filename):
        """Read a list of word tag classes"""
        with open(filename, 'r') as f:
            tags = f.read().split("\n")
        return tags

    # subclass Line by providing data as format: word1 tag1 \n word2 tag2
    class Line:
        def __init__(self, line):
            words = line.split("\n")
            self.id = words[0]  # first part is id of the sentence
            self.words = []
            self.tags = []

            for idx in range(1, len(words)):
                # each pair is separated by a tab
                pair = words[idx].split("\t")

                # add word to word dict
                self.words.append(pair[0])

                # add tag to tag dict
                self.tags.append(pair[1])

            # length of the line
            self.length = len(self.words)
            return

        # show the content of the Line
        def show(self):
            print(self.id)
            print(self.length)
            print(self.words)
            print(self.tags)
            return

# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # Edit here
    """
    Here WORD is observation and TAG is state
    """

    # Obtain state_dict
    state_dict = {}
    obs_dict = {}
    idxWord = 0

    # create the state_dict
    for idxTag, tag in enumerate(tags):
        state_dict[tag] = idxTag
    # print("state_dict:", state_dict)

    # number of states
    S = len(state_dict.keys())
    # print("number of tags: ", S)

    # create initial probability Pi vector
    pi = np.zeros(S, dtype=float)
    A = np.zeros((S, S), dtype=float)

    # loop through each sentence
    for l in train_data:
        # get the initial tag of each sentence
        firstTag = l.tags[0]

        # update initial probability Pi vector
        pi[state_dict[firstTag]] += 1.0

        # loop through each word
        for word in l.words:
            # if word not in obs_dict yet, add to it
            if word not in obs_dict:
                obs_dict[word] = idxWord
                idxWord += 1

        # loop through each tag
        for idxTag, tag in enumerate(l.tags[:-1]):
            # update the transition prob matrix
            nextTag = l.tags[idxTag+1]

            # update A matrix
            A[state_dict[tag]][state_dict[nextTag]] += 1

    # print("obs_dict:", obs_dict)

    # normalize initial probability
    sumPi = sum(pi)
    pi = [x/sumPi for x in pi]
    # print("pi:", pi)

    # normalize to avoid zeros
    A += 1

    # normalize A by total sum along each row to get probability
    # print("A:", A)
    # print(np.sum(A, axis=1))
    A = A/(np.sum(A, axis=1).reshape(S, 1))
    # print("A:", A)
    # print(sum(A[0]))

    # number of observation symbols
    N = len(obs_dict.keys())
    # print("number of words: ", N)

    # initialize emission probability matrix
    B = np.zeros((S, N), dtype=float)
    for l in train_data:
        for idxTag, tag in enumerate(l.tags):
            # word with corresponding tag
            word = l.words[idxTag]

            # update B matrix
            B[state_dict[tag]][obs_dict[word]] += 1

    # normalize to avoid zeros
    B += 1

    # normalize A by total sum along each row to get probability
    # print("B:", B)
    # print(np.sum(B, axis=1))
    B = B/(np.sum(B, axis=1).reshape(S, 1))
    # print("B:", B)
    # print(sum(B[0]))

    # create model from training data
    model = HMM(pi, A, B, obs_dict, state_dict)
    ###################################################
    return model


# TODO:
def speech_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    # Edit here
    # loop through each line
    for l in test_data:
        # get the corresponding observation sequence a numpy array of actual observation WORDS
        wordArray = np.asarray(l.words)
        # print("word array:", wordArray)

        # perform viterbi decoding
        stateSequence= model.viterbi(wordArray)
        # print("state sequence index: ", stateSequence)

        # update final tagging
        tagging.append(stateSequence)

    ###################################################
    return tagging
