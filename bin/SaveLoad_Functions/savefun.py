import _pickle as cpickle
import os

def saveModel(model_name, classifier, method):
    path = ("G:\\Casper\\GitKraken\\srp\\Saves\\Models\\{}\\".format(method))
    with open( path + '{}'.format(model_name), 'wb') as picklefile:
        cpickle.dump(classifier, picklefile)

def exportResult(path, name, method, cutoff, doclen, max_features, min_df, max_df, norm, test_size, n_estimators, max_depth, confmatrix, classreport, accscore, passedTime):
    outfile = open(path + '\\{}.txt'.format(name), 'w')
    outfile.write(
    """Method: {0} \n
Cutoff: {1} \n
Number of documents: {2} \n \n
Vectorizer settings: \n
Max features: {3} \n
Min df: {4} \n
Max df: {5} \n 
Normalization: {6} \n \n
Data Split Ratio: {7} \n \n
Number of trees in the forest: {8} \n 
Maximum depth of tree: {9} \n \n
Confusion Matrix:\n {10}\n 
Classreport:\n {11}\n 
Accuracy Score:\n {12}\n
Time passed: {13} seconds"""
    .format(
        method,
        cutoff,
        doclen,
        max_features,
        min_df,
        max_df,
        norm,
        test_size,
        n_estimators,
        max_depth,
        confmatrix, 
        classreport, 
        accscore,
        round(passedTime, 4)))

def saveCleanedData(folder_name, tokens, labels):
    path = 'G:\\Casper\\GitKraken\\srp\\Saves\\Cleaned Data\\{}'.format(folder_name)
    if not os.path.exists(path):
        os.mkdir(path)
    with open (path + '\\tokens', 'wb') as cleaned_tokens:
        cpickle.dump(tokens, cleaned_tokens)
    with open (path + '\\labels', 'wb') as cleaned_labels:
        cpickle.dump(labels, cleaned_labels)