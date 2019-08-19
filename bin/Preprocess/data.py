import json
import logging as log
from collections import Counter
from nltk.corpus import stopwords
import nltk

def import_data(filename):
    with open(filename) as json_data:
        data = json.load(json_data)

    row_id = []
    subject_id = []
    category = []
    description = []
    words = []

    for line in data:
        row_id.append(line['ROW_ID'])
        subject_id.append(line['SUBJECT_ID'])
        category.append(line['CATEGORY'])
        description.append(line['DESCRIPTION'])
        text = line['TEXT']
        text = text.split()
        # text = nltk.word_tokenize(text)
        words.append(text)
    
    unique_cat = set(category)
    unique_des = set(description)

    cat = [Counter(category)]        
    
    print("Data contains %s row id's, %s subject id's, and %s text fields" % (len(row_id), len(subject_id), len(words)))
    print("There are %s unique categories, and %s unique descriptions" % (len(unique_cat), len(unique_des)))
    print('List of categories: ', cat)

    # listofcat = []
    # listofcat.append(unique_cat)
    # for i in range(len(unique_cat)):
    #     for cat in unique_cat:
    #         listofcat[i].append(category.count(cat))

    return row_id, subject_id, category, description, words

def clean_data(text):
    #Strip date
    x = 0
    for i in range(len(text) - 1, -1, -1):
        #Strip symbols
        for j in range(len(text[i]) - 1, -1, -1):
            text[i][j] = text[i][j].replace(':', '')
            text[i][j] = text[i][j].replace('#', '')
    # for text[i] in text:       # Loop through all text fields
    #     for word in text[i]:   # Loop through one text field
            # log.info('text[i]: {0}, word: {1}'.format(text[i],word))
            if text[i][j].startswith('[') or text[i][j].endswith(']'):
                del text[i][j]
                x += 1
                continue
            if text[i][j].startswith('(') and text[i][j].endswith(')'):
                del text[i][j]
                x += 1
                continue
            if text[i][j].startswith('__') and text[i][j].endswith('__'):
                del text[i][j]
                x += 1
                continue
            if '*' in text[i][j]:
                del text[i][j]
                x += 1
                continue
        
        while '' in text[i]:
            text[i].remove('')
            x += 1
   
    print('Removed %s entries' % (x))

def removeStopword(text):
    x = 0
    stopWords = set(stopwords.words('english'))
    for i in range(len(text) - 1, -1, -1):
        for j in range(len(text[i]) - 1, -1, -1):
    # for lst in text:
        # for item in lst:
            if text[i][j] in stopWords:
                del text[i][j]
                x += 1
    print('Removed %s stopwords.' % (x))
