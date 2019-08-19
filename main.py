#Stackabuse version 0.10.0
import os
import json
import random
import re
from nltk.tokenize import PunktSentenceTokenizer
import bin.Preprocess.data as d
import bin.Preprocess.cutdata as cd
import bin.Preprocess.cleandatafun as cdf
import bin.BagofWords as bg
import bin.SaveLoad_Functions.loadfun as lf
import bin.SaveLoad_Functions.savefun as sf
import bin.Visualize_functions as vis
import time
import _pickle as cpickle
import matplotlib.pyplot as plt
import gc
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# row_id, subject_id, category, description, words = d.import_data('file.json')
# d.clean_data(words)
# d.removeStopword(words)

######## Save small set of shuffled data ##############
# data = cd.getData()
# dat = data[:1000]
# cd.saveJson('file.json', dat)

# document = str(words)
# sentences = nltk.sent_tokenize(document)
# for sent in sentences:
#         print(nltk.pos_tag(nltk.word_tokenize(sent)))

# ONLY LEAVE ONE METHOD UNCOMMENTED!!!
# method = 'RandomForestClassifier'
# method = 'DecisionTreeClassifier'
# method = 'SGDClassifier'
# method = 'SVCpoly'
method = 'SVCrbf'

'''PARAMS'''
cutoff = 100       #integer
method_path = ('G:\\Casper\\GitKraken\\srp\\Saves\\Results\\{}'.format(method))
#Vectorizer
max_features = 750   #integer
min_df = 5         #integer
max_df = 0.7        #ratio
norm = 'l2'
# #Data Splitting
test_size = 0.2     #ratio
# #Model Fitting
n_estimators = 20    #integer (number of trees in the forest)
max_depth = 20      #integer (tree depth)

# def runMain(cutoff,max_features, min_df, max_df, n_estimators, max_depth):
start = time.time()
cutoff_path = (method_path + '\\{}'.format(cutoff))

if not os.path.exists(method_path):
    os.mkdir(method_path)
if not os.path.exists(cutoff_path):
    os.mkdir(cutoff_path)

'''LOADING DATA '''
# data, labels = lf.loadData()
documents, labels, legend = lf.loadPickledata(cutoff) #occurrence
if not os.path.exists(cutoff_path + '\\0datastructure.png'):
    vis.visualizeData(labels, legend, cutoff_path, True)

# '''CLEANING DATA '''
# # documents = cdf.cleanData(data)
# # tokens = cdf.spacyClean(data)

# '''LOADING MODELS'''
# # labels = lf.loadCleanedData('1000')
# # tokens = lf.loadCleanedData('1000')

''' PIPELINE '''
# grid_search = bg.gridSearch(documents, labels, cutoff, classifier= 2, save=True)

# '''EXPERIMENT'''
#vectors, features = bg.vectorize(documents, max_features, min_df, max_df)
# # vectors = bg.vectorize(tokens)
# tfid = bg.tfid(vectors)
tfid, features, vectorizer= bg.vectTfidf(documents, max_features, min_df, max_df, norm)


# tsne = TSNE(n_components=2, random_state=42)
# print('t-SNE transform...')
# X_2d = tsne.fit_transform(tfid)
# print('Done!')

# X_train, X_test, y_train, y_test, doc_train, doc_test = bg.dataSplit(tfid, documents, labels, test_size)   #3rd variable for test size (default 0.2)
# X_train, X_test, y_train, y_test, doc_train, doc_test = bg.dataSplit(X_2d, documents, labels, test_size)   #3rd variable for test size (default 0.2)
# # X_2d_train = tsne.fit_transform(X_train)
# print('tsne...')
# X_2d_test = tsne.fit_transform(X_test)
# print('tsne done')
# # # del tfid
# classifier = bg.fitModel(X_train, y_train, n_estimators, method, max_depth)
# # # del X_train
# # # del y_train
# # # classifier.feature_importances_
# # # ytest, predictions = bg.runKfold(tfid, labels)
# y_pred = bg.predictModel(X_test, classifier)
# confmatrix, classreport, accscore = bg.evaluateModel(y_test, y_pred)
# # # scores = bg.validateModel(classifier, tfid, labels) #Not needed as it trains the model twice

# # # ''' SAVE '''
# # # # sf.saveCleanedData(cutoff, documents, labels)
# # # sf.saveModel(cutoff, classifier, method)

if not os.path.exists(cutoff_path + '\\{}'.format(round(accscore,4))):
    os.mkdir(cutoff_path + '\\{}'.format(round(accscore,4)))
result_path = cutoff_path + '\\{}'.format(round(accscore,4))
iteration = 1  
for root, dirs, files in os.walk(result_path):
    files.sort(key=natural_keys)
    for item in files:
        y = item[0]
        z = y + item[1]
        if int(y) == 9:
            iteration = 10
        elif iteration >= 10:
            while iteration == int(z):
                iteration += 1
        else:
            while iteration == int(y):
                iteration += 1
print('Creating plots...')
plt.close(fig='all')
passedTime = time.time() - start
# vis.visualizeDecisionBoundary(X_2d, labels, classifier, tfid, accscore, method, save=True)
# sf.exportResult(result_path, str(iteration), method, cutoff, len(documents), max_features, min_df, max_df, norm, test_size, 
                # n_estimators, max_depth, confmatrix, classreport, accscore, passedTime)
# # vis.visualizeFeatures(features, classifier, path=result_path, name=str(iteration), save=True)
# vis.visualizeDataSplit(y_train, y_test, legend, path=result_path, name=str(iteration), save=True)
# vis.visualizeConfusion(confmatrix, accscore, legend, normalize=True, path=result_path, name=str(iteration), save=True)
# # vis.visualizeConfusion(confmatrix, accscore, legend, normalize=False, path=result_path, name=str(iteration), save=True)
# vis.visualizeFeatSpace(tfid, classifier, labels, legend, features, shrink=False, path=result_path, name=str(iteration), save=True)
# # vis.visualizeTree(classifier, features, labels, legend, path=result_path, name=str(iteration))
# vis.reviewDoc(y_pred, y_test, doc_test, legend, path=result_path, name=str(iteration))


'''
i = 0
while i < 15 :
    print('Starting iteration {}'.format(i))
    documents, labels, legend = lf.loadPickledata(cutoff)
    tfid, features, vectorizer= bg.vectTfidf(documents, max_features, min_df, max_df, norm)
    labelid = i
    vis.visualizeTSNEsingle(tfid, labels, legend, cutoff, max_features, labelid, name = str(i), save = True)
    i += 1
    print('Deleting variables')
    del documents, labels, legend, tfid, features, vectorizer
    print('Garbage collection')
    gc.collect()
passedTime = time.time() - start
'''

print('Time passed: {} seconds'. format(round(passedTime, 4)))