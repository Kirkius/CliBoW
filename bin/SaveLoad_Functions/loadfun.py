import _pickle as cpickle
from sklearn.datasets import load_files
import numpy as np
from random import shuffle
from collections import Counter
import os

def loadData():
    text_data = load_files(r'G:\Casper\GitKraken\srp\data\data(sample)')

    return text_data.data, text_data.target

def loadPickledata(cutoff):
    data = []
    labels = []
    # string = ""
    legend = {}
    # occurrence = {}
    i = 0
    for roots, dirs, files in os.walk('G:\\Casper\\GitKraken\\srp\\Saves\\Cleaned Pickle'):
        files = files
        print('Loaded filelist')
    for pickles in files:
        with open('G:\\Casper\\GitKraken\\srp\\Saves\\Cleaned Pickle\\%s' % (pickles), 'rb') as pickle:
            temp = (cpickle.load(pickle))
        if len(temp) > cutoff:
            shuffle(temp)
            temp = temp[:cutoff]
            print('Reduced length of {} to {} entries'.format(pickles, cutoff))
        print('Labeling {} with {}'.format(pickles, i))
        lbl = np.full((len(temp)), i)
        labels = np.concatenate((labels, lbl))
        # for doc in temp:
            # string = string + doc
        # string = string.split()
        # count = Counter(string)
        legend[i] = "{}".format(pickles)
        # occurrence[i] = count.most_common(10)
        i += 1
        string = ""
        data = data + temp
    
    return data, np.asarray(labels, dtype=np.uint8), legend#, occurrence

def loadModel(model_name):
    with open('G:\\Casper\\GitKraken\\srp\\Saves\\Models\\%s' % (model_name), 'rb') as training_model:
        model = cpickle.load(training_model)
    
    return model

def loadPickle(model_name):
    with open('G:\\Casper\\GitKraken\\srp\\Saves\\Pickle\\%s' % (model_name), 'rb') as training_model:
        model = cpickle.load(training_model)
    
    return model

def loadCleanedData(folder_name):
    path = 'G:\\Casper\\GitKraken\\srp\\Saves\\Cleaned Data\\{}'.format(folder_name)
    with open (path + '\\tokens', 'rb') as cleaned_tokens:
        tokens = cpickle.load(cleaned_tokens)
    with open (path + '\\labels', 'rb') as cleaned_labels:
        labels = cpickle.load(cleaned_labels)
    
    return tokens, labels
