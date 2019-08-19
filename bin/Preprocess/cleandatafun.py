import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import time
import string
from nltk.stem.porter import PorterStemmer
import _pickle as cpickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

def cleanData(data):
    documents = []
    stemmer = WordNetLemmatizer()
    # totalTimeFolder = 0
    # start = 0
    n_iter = len(data)
    for i in range(n_iter):
        # start = time.time()
        document = str(data[i])
        last = ''
        while document != last:
            last = document
            document = re.sub(r'[:*\[\]\'/#\(\)\{\}\"_]?(\\r\\n)?(b\')?(, )?(\. )?(\\\\)?', '', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'(\s+)?( - )', ' ', document, flags=re.I)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    #     end = time.time()
    #     passedTime = end - start   
    #     totalTimeFolder = totalTimeFolder + passedTime
    #     avgTime = totalTimeFolder / (i + 1)
    #     remainingTime = (avgTime * float(n_iter)) - totalTimeFolder
    #     percent = (i+1) / n_iter * 100
    #     print("Cleaned {0} / {1} documents. {3}% Estimated time left: {5} Seconds. Average time: {4}s. Read time: {2}s."
    #         .format(
    #             i+1,
    #             n_iter,
    #             round(passedTime, 4),
    #             round(percent, 2),
    #             round(avgTime, 2),
    #             int(round(remainingTime, 0))),
    #         end="          \r"),
    # print("")
    
    return documents

def spacyClean(data):
    print('Loading Spacy model...')
    nlp = spacy.load('en_core_web_md')
    spacydocs = []
    tokens = []
    print('Done!')
    print('Applying Spacy model. This might take a while...')
    totalTimeFolder = 0
    start = 0
    n_iter = len(data)
    for i in range(n_iter):
        start = time.time()
        spacydoc = str(data[i])
        spacydoc = nlp(spacydoc)
    # spacydocs = nlp.pipe(str(data), batch_size = 1000, n_threads = -1)
        spacydocs.append(spacydoc)
        end = time.time()
        passedTime = end - start   
        totalTimeFolder = totalTimeFolder + passedTime
        avgTime = totalTimeFolder / (i + 1)
        remainingTime = (avgTime * float(n_iter)) - totalTimeFolder
        percent = (i+1) / n_iter * 100
        print("Cleaned {0} / {1} documents. {3}% Estimated time left: {5} Seconds. Average time: {4}s. Read time: {2}s."
            .format(
                i+1,
                n_iter,
                round(passedTime, 4),
                round(percent, 2),
                round(avgTime, 2),
                int(round(remainingTime, 0))),
            end="          \r"),
    print("")
    print('Spacydoc created!')
    # spacydoc = list(spacydoc) 
    print('Start Tokenizing...')
    totalTimeFolder = 0
    start = 0
    n_iter = len(spacydocs)
    for j in range(n_iter):
        start = time.time()
        token = [w.text for w in spacydocs[j] if w.is_alpha or w.is_digit]
        token = ' '.join(token)
        tokens.append(token)
        passedTime = end - start   
        totalTimeFolder = totalTimeFolder + passedTime
        avgTime = totalTimeFolder / (j + 1)
        remainingTime = (avgTime * float(n_iter)) - totalTimeFolder
        percent = (j+1) / n_iter * 100
        print("Cleaned {0} / {1} documents. {3}% Estimated time left: {5} Seconds. Average time: {4}s. Read time: {2}s."
            .format(
                j+1,
                n_iter,
                round(passedTime, 4),
                round(percent, 2),
                round(avgTime, 2),
                int(round(remainingTime, 0))),
            end="          \r"),
    print("")  
    return tokens

def nltkClean(data):
    temp = []
    tokens = []
    table = str.maketrans('', '', string.punctuation)
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    for x in range(len(data)):
        temp.append(data[x].decode())
    for x in range(len(temp)):
        tokens.append(word_tokenize(temp[x]))
        tokens[x] = [w.lower() for w in tokens[x]]
        tokens[x] = [w.translate(table) for w in tokens[x]]
        tokens[x] = [word for word in tokens[x] if word.isalpha()]
        tokens[x] = [w for w in tokens[x] if not w in stop_words]
        tokens[x] = [porter.stem(word) for word in tokens[x]]
    return tokens

# with open('G:\\Casper\\GitKraken\\srp\\Saves\\Pickle\\Case Management', 'rb') as pickle:
#             data = (cpickle.load(pickle))

# tokens = nltkClean(data)