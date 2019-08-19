import numpy as np
import re
import nltk
from sklearn.datasets import load_files
# nltk.download('stopwords')
# nltk.download('wordnet')
import pickle
import spacy
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from pactools.grid_search import GridSearchCVProgressBar
import warnings
import joblib
from sklearn.svm import SVC

def vectorize(documents, max_features, min_df, max_df):
    """ Vectorize data according to the Bag-of-Words model

    Parameters
    ----------
    documents : array-like
        Data to be vectorized. Data must be a list containing lists of documents

    max_features : int
        Determines how many features are passed to the classifier

    min_df : int
        Determines minimum amount of documents a word must be present in to be considered a feature
    
    max_df : float
        Determines the maximum percentage of documents a word may be present in to still be considered a feature
    
    Returns
    -------
    X : array, [n_samples, n_features]
        Document-term matrix.
    
    feature_names : array
        Mapping from feature integer indices to feature name
    """
    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, stop_words=stopwords.words('english'))
    print('Vectorizing data...')
    X = vectorizer.fit_transform(documents).toarray()
    feature_name = vectorizer.get_feature_names()
    print(X.shape)

    return X, feature_name

def tfid(vectors):
    """ Converts document-term matrix data to Term Frequency - Inverse Document Frequency (TF-IDF) array

    Parameters
    ----------
    vectors : array-like
        Document-term matrix data from the vectorizer

    Returns:
    ----------
    tfidf : array-like
        Array containing Term Frequency data rescaled to Inverse Document Frequency
    """
    tfidconverter = TfidfTransformer()
    print('Converting to TF-IDF...')
    tfid = tfidconverter.fit_transform(vectors).toarray()

    return tfid

def vectTfidf(documents, max_features, min_df, max_df, norm):
    '''
    Combines the vectorize and tfid functions

    Parameters
    ----------
    documents : array-like
        Data to be vectorized. Data must be a list containing lists of documents

    max_features : int
        Determines how many features are passed to the classifier

    min_df : int
        Determines minimum amount of documents a word must be present in to be considered a feature
    
    max_df : float
        Determines the maximum percentage of documents a word may be present in to still be considered a feature
    
    norm : 'l1' or 'l2'
        Either 'l2' Sum of squares vector, the cosine similairity between two vectors is their dot product, or 'l1' Sum of absolute values of vector elements

    Returns:
    ----------
    tfidf : array-like
        Array containing Term Frequency data rescaled to Inverse Document Frequency

    '''
    # Lambda function to override the preprocessor and tokenizer module. Input data already cleaned and tokenized
    vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, max_features=max_features, min_df=min_df, max_df=max_df, lowercase=False, norm=norm)
    print('Vectorizing data and converting to TF-IDF...')
    tfid = vectorizer.fit_transform(documents).toarray()
    feature_name = vectorizer.get_feature_names()

    return tfid, feature_name, vectorizer

def dataSplit(data, docs, labels, test_size):
    '''
    Shuffles and splits the data, labels, and raw documents in a train and test according to given ratio

    Parameters
    ----------
    data : array-like
        Data to be split

    docs : list
        Raw documents to be split

    labels : list
        List of labels to be split

    test_size : float
        Ratio along which to split the data

    Returns:
    ----------
    X_train
    X_test
    y_train
    y_test
    doc_train
    doc_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    doc_train, doc_test = train_test_split(docs, test_size=test_size, random_state=42)
    print('Data split with a {} ratio'.format(test_size))
    return X_train, X_test, y_train, y_test, doc_train, doc_test

def fitModel(train_data, train_labels, n_estimators, method, max_depth=None):
    if method == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state=42)
    elif method == 'DecisionTreeClassifier':
        classifier = DecisionTreeClassifier(random_state=42)
    elif method == 'SGDClassifier':
        classifier = SGDClassifier(class_weight='balanced', n_jobs= -1, shuffle = True, max_iter= 500, random_state=42)
    elif method == 'SVCpoly':
        classifier = SVC(C=1, kernel='poly', class_weight='balanced', gamma='scale', random_state=42)
    elif method == 'SVCrbf':
        classifier = SVC(C=1, kernel='rbf', class_weight='balanced', gamma='scale', random_state=42)
    print('Fitting model using {} method...'.format(method))
    classifier.fit(train_data, train_labels)

    return classifier

def predictModel(test_data, classifier):
    print('Predicting...')
    y_pred = classifier.predict(test_data)
    
    return y_pred

def runKfold(data, labels, n_folds=3, n_estimators=1000):
    kf = KFold(n_splits=n_folds, shuffle=True)
    outcomes = []
    fold = 0
    classifier = RandomForestClassifier(n_estimators, random_state=42)
    for train_index, test_index in tqdm(kf.split(data)):
        fold += 1
        Xtrain, Xtest = data[train_index], data[test_index]
        ytrain, ytest = labels[train_index], labels[test_index]
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)
        accuracy = accuracy_score(ytest, predictions)
        outcomes.append(accuracy)
        # print(train_index, test_index)
    mean_outcome = np.mean(outcomes)
    print(outcomes)
    print('Mean Accuracy: {}'.format(mean_outcome))
    return ytest, predictions

def evaluateModel(test_labels, pred_data):
    confmatrix = confusion_matrix(test_labels, pred_data)
    classreport = classification_report(test_labels, pred_data)
    accscore = accuracy_score(test_labels, pred_data)
    print('Confusion Matrix:')
    print(confmatrix)
    print('Classification Report:')
    print(classreport)
    print('Accuracy Score: %s' % (accscore))

    return confmatrix, classreport, accscore

def validateModel(classifier, data, labels):
    scores = cross_val_score(classifier, data, labels, cv = 3)
    # kf = KFold(n_splits=3)
    # for train_index, test_index in kf.split(data):
    #     x_train, y_train = data[train_index], labels[train_index]
    #     x_test, y_test = data[test_index], labels[test_index]

    return scores

def gridSearch(documents, labels, cutoff, classifier = 0, save = False):
    warnings.filterwarnings("ignore")
    if classifier == 0:
        pipeline = Pipeline([
            ('tfid', TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, lowercase=False)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        parameters = {
        'tfid__max_features': (100, 250, 500, 750),
        'tfid__min_df': (5,10, 20, 30),
        'tfid__max_df': (0.7, 0.8, 0.9),
        'classifier__n_estimators': (10, 20, 30),
        'classifier__max_depth': (5, 15, 25, None)
        }
    elif classifier == 1:
        pipeline = Pipeline([
            ('tfid', TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, lowercase=False)),
            ('classifier', SGDClassifier(class_weight = 'balanced', random_state=42))
        ])
        parameters = {
        'tfid__max_features': (100, 250, 500, 750),
        'tfid__min_df': (5,10, 20, 30),
        'tfid__max_df': (0.7, 0.8, 0.9),
        'classifier__shuffle': (True, False),
        'classifier__max_iter': (500, 1000, 1500)    
        }
    elif classifier == 2:
        pipeline = Pipeline([
            ('tfid', TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, lowercase=False, max_features=750, min_df=5, max_df=0.7)),
            ('classifier', SVC(class_weight='balanced', random_state=42, gamma='scale'))
        ])
        parameters = {        
        'classifier__C': (0.001, 0.01, 0.1, 1, 10),
        'classifier__kernel': ('rbf', 'poly', 'linear')
        }
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=3, verbose=3, n_jobs=-1, return_train_score=True)
    grid_search_tune.fit(documents, labels)
    print('Best parameters set:')
    print(grid_search_tune.best_params_)
    print('Best score:')
    print(grid_search_tune.best_score_)
    if save:
        outfile = open('G:\\Casper\\GitKraken\\srp\\Saves\\Results\\Gridsearch\\{}.txt'.format(str(round(grid_search_tune.best_score_,4))), 'w')
        outfile.write(
            '''
Cutoff: {0} \n \n
Used parameters: \n
{1} \n \n
Best parameters: \n
{2} \n \n
Best score: \n
{3}
            '''
            .format(cutoff,
            parameters, 
            grid_search_tune.best_params_,
            grid_search_tune.best_score_)
            )
        # joblib.dump(grid_search_tune, 'G:\\Casper\\GitKraken\\srp\\Saves\\Results\\Gridsearch\\{}.pkl'.format(str(round(grid_search_tune.best_score_,4))))
    else:
        return grid_search_tune

