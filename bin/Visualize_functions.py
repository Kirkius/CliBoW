import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import texttable as tt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.tree import export_graphviz
from subprocess import call
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from matplotlib.lines import Line2D
from mlxtend.plotting import plot_decision_regions
# from IPython import Image

def visualizeData(labels, legend, path=None, save=False):
    plt.close()
    plt.cla()
    plt.figure(figsize=(9,5))
    plt.subplots_adjust(bottom=0.3, top = 0.99)
    lbl, freq = np.unique(labels, return_counts=True)
    y_pos = np.arange(len(lbl))
    bar1 = plt.bar(y_pos, freq)
    lgd = []
    for x in lbl:
        lgd.append(legend[x])
    plt.xticks(y_pos, lgd, rotation=45, rotation_mode='anchor', ha='right')
    plt.ylabel('Frequency of label')
    plt.xlabel('Labels')
    for i in bar1:
        height = i.get_height()
        plt.text(i.get_x() + i.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    if save:
        plt.savefig(path + '\\0datastructure.png')
    else:
        plt.show()

def visualizeDataSplit(y_train, y_test, legend, classifier, path=None, name=None, save=False):  
    plt.close()
    plt.cla()
    lbl_train, freq_train = np.unique(y_train, return_counts=True)
    lbl_test, freq_test = np.unique(y_test, return_counts=True)
    y_pos = np.arange(len(lbl_train))
    plt.figure(figsize=(15,8))
    plt.subplots_adjust(bottom=0.15, top = 0.99)
    lgd = []
    for x in lbl_train:
        lgd.append(legend[x])
    bar_train = plt.bar(y_pos-0.2, freq_train,width=0.4,color='#4444ff',align='center')
    for i in bar_train:
        height = i.get_height()
        plt.text(i.get_x() + i.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    bar_test = plt.bar(y_pos+0.2, freq_test,width=0.4,color='#ff4444',align='center')
    for i in bar_test:
        height = i.get_height()
        plt.text(i.get_x() + i.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    plt.xticks(y_pos, lgd, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Train/Test Split')
    plt.ylabel('Frequency of label')
    plt.legend(['Train Data', 'Test Data'])
    if save:
        plt.savefig(path + '\\{}DataSplit.png'.format(name))
    else:
        plt.show()

def visualizeFeatures(features, classifier, path=None, name=None, save=False): 
    plt.close()
    plt.cla()
    importances = classifier.feature_importances_
    indices = np.argsort(-importances)
    topIndices = indices[:20]
    featureArray = np.array(features)
    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(topIndices)), importances[topIndices], color='b', align='center')
    plt.yticks(range(len(topIndices)), featureArray[topIndices])
    plt.xlabel('Relative Importance')
    plt.subplots_adjust(right = 0.99, left = 0.18)
    if save:
        plt.savefig(path + '\\{}features.png'.format(name))
    else:
        plt.show()

def visualizeConfusion(confmatrix, accscore, legend, normalize=False, path=None, name=None, save=False):
    plt.close()
    plt.cla()
    val = []
    for x in legend:
        val.append(legend[x])
    if normalize:
        cm = confmatrix.astype('float') / confmatrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12,9))
        sns.heatmap(cm, annot=True, fmt=".0%", linewidths=1, square = False, cmap = 'RdBu_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.xticks(range(len(val)), val, rotation = 45, rotation_mode = 'anchor', ha = 'right')
        plt.yticks(range(0, len(val)), val, rotation = 'horizontal' , rotation_mode = 'anchor', va = 'top')
        all_sample_title = 'Accuracy Score: {}'.format(accscore)
        plt.title(all_sample_title, size = 15)
        plt.subplots_adjust(bottom = 0.16, top = 0.95, right = 1, left = 0.15)
        if save:
            plt.savefig(path + '\\{}confusionmatrix[normalized].png'.format(name))
        else:
            plt.show()
    else:
        plt.figure(figsize=(12,9))
        sns.heatmap(confmatrix, annot=True, fmt="d", linewidths=1, square = False, cmap = 'RdBu_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.xticks(range(len(val)), val, rotation = 45, rotation_mode = 'anchor', ha = 'right')
        plt.yticks(range(0, len(val)), val, rotation = 'horizontal' , rotation_mode = 'anchor', va = 'top')
        all_sample_title = 'Accuracy Score: {}'.format(accscore)
        plt.title(all_sample_title, size = 15)
        plt.subplots_adjust(bottom = 0.16, top = 0.95, right = 1, left = 0.15)
        if save:
            plt.savefig(path + '\\{}confusionmatrix.png'.format(name))
        else:
            plt.show()

def visualizeOccurrence(occurrence, legend):
    tab = tt.Texttable()
    tab.header(legend)
    #occurence to np array
    t = np.array(list(occurrence.items()))
    #only get second index
    t = t[:, 1]
    #convert second dimension to np array
    t = [np.array(i) for i in t]
    #transpose first and second dimension
    t = np.swapaxes(t, 0, 1)
    for i in legend:
        tab.add_row(t[i])
    print(tab.draw())

def visualizeFeatSpace(tfid, classifier, labels, legend, features, num_feat=5, shrink=False, n_samples=5, path=None, name=None, save=False):
    importances = classifier.feature_importances_
    indices = np.argsort(-importances)
    topIndices = indices[:num_feat]
    featureArray = np.array(features)
    topfeatures = featureArray[topIndices]
    labelnames = []
    markers = ['o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X']
    for x in labels:
        labelnames.append(legend[x])
    t = tfid[:,topIndices]
    df = pd.DataFrame(t, columns=topfeatures)
    dfDirty = df+0.00001*np.random.rand(len(labels), num_feat) #Introducing noise is required to cluster on labels
    dfDirty['label'] = pd.Series(labelnames, index=df.index)
    if shrink:
        dfDirty = dfDirty.groupby('label').apply(lambda x: x.sample(n_samples)).reset_index(drop=True) #Takes 'n_samples' random samples from each class
    sns.pairplot(dfDirty, hue = 'label', diag_kind='kde', markers=markers, plot_kws={'alpha':0.5})
    if save:
        plt.savefig(path + '\\{}pairplot.png'.format(name))
    else:
        plt.show()

def visualizeTSNE(tfid, labels, legend, cutoff, max_features, name = None, shrink=False, save=True):
    plt.close()
    plt.cla()
    plt.figure(figsize=(16,9))
    tsne = TSNE(n_components=2, random_state=42)
    target_ids = range(len(set(labels)))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'xkcd:violet blue', 'orange', 'purple', 'xkcd:rose pink', 'xkcd:aqua green', 'xkcd:deep sea blue', 'xkcd:grey purple', 'xkcd:dandelion'
    markers = ['o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X']
    labelnames = []
    for x in set(labels):
        labelnames.append(legend[x])
    if shrink: #This shrink option oversamples classes with few documents, creating an inaccurate graph
        df = pd.DataFrame(tfid)
        df['label'] = pd.Series(labels, index=df.index)
        dfReduced = df.groupby('label').apply(lambda x: x.sample(int(cutoff/2), replace=True)).reset_index(drop=True) #Samples half of all datapoints per class max
        labels = dfReduced['label'].values
        tfidReduced = dfReduced.values
        tfidReduced = np.delete(tfidReduced, max_features, 1) #deletes the last column containing the labels
        X_2d = tsne.fit_transform(tfidReduced)
        plt.title('TSNE-Reduced(number of datapoints = {0} / orignal = {1}'.format(len(tfidReduced),len(tfid)))
    else:
        X_2d = tsne.fit_transform(tfid)
        plt.title('TSNE(number of datapoints = {})'.format(len(tfid)))
    for i, c, label in zip(target_ids, colors, labelnames):
        plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c = c, label = label, marker=markers[i], alpha=0.7)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    if save:
        plt.subplots_adjust(right=0.86)
        plt.savefig('G:\\Casper\\GitKraken\\srp\\Saves\\Results\\t-SNE\\{0}\\{1}.png'.format(cutoff, name))
    else:
        plt.subplots_adjust(right=0.86)
        plt.show()

def visualizeTSNEsingle(tfid, labels, legend, cutoff, max_features, labelid, name = None, save = False):
    plt.close()
    plt.cla()
    plt.figure(figsize=(16,9))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'xkcd:violet blue', 'orange', 'purple', 'xkcd:rose pink', 'xkcd:aqua green', 'xkcd:deep sea blue', 'xkcd:grey purple', 'xkcd:dandelion'
    markers = ['o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X']
    #clr and mrk ensure the same color and symbol is used for the class as in the "whole" t-SNE
    clr = colors[labelid]
    mrk = markers[labelid]
    tsne = TSNE(n_components=2, random_state=42)
    #Convert tfid matrix into dataframe
    df = pd.DataFrame(tfid)
    #Add the labels as a column to the dataframe
    df['labels'] = labels
    #Select only the row which have the same label as the labelid
    dfsingle = df.loc[df['labels'] == labelid]
    #Create new variable with only the labels matching the labelid
    singlelabel = dfsingle['labels']
    #Remove the labels column from the dataframe
    dfsingle = dfsingle.drop(['labels'], axis = 1)
    X_2d = tsne.fit_transform(dfsingle)
    plt.title('t-SNE on a single class (number of datapoints = {})'.format(len(dfsingle.index)))
    plt.scatter(X_2d[singlelabel == labelid, 0], X_2d[singlelabel == labelid, 1], c = clr, label = legend[labelid], marker = mrk, alpha = 0.7)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    if save:
        plt.subplots_adjust(right=0.86)
        plt.savefig('G:\\Casper\\GitKraken\\srp\\Saves\\Results\\t-SNE\\single\\{0}.png'.format(name))
    else:
        plt.subplots_adjust(right=0.86)
        plt.show()

def interactiveTSNE(tfid, labels, legend, cutoff, max_features, documents, labelid=None):
    plt.close()
    plt.cla()
    def onclick(event):
        import numpy as np
        line = event.artist
        ind = event.ind
        xy = line.get_offsets()
        if len(ind) > 1:
            print('Mutliple points...')
            for i in ind:
                point = xy[i]
                data = np.where(X_2d == point)
                doc = data[0][0]
                label = labels[doc]
                print('Label: {}'.format(legend[label]))
                print('Document:')
                print(documents[doc])
                print()
                input("Press Enter to continue...")
                print()
        else:
            point = xy[ind]
            data = np.where(X_2d == point)
            doc = data[0][0]
            label = labels[doc]
            print('Label: {}'.format(legend[label]))
            print('Document:')
            print(documents[doc])
            print()           
    fig, ax = plt.subplots()
    # plt.figure(figsize=(16,9))
    tsne = TSNE(n_components=2, random_state=42)
    target_ids = range(len(set(labels)))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'xkcd:violet blue', 'orange', 'purple', 'xkcd:rose pink', 'xkcd:aqua green', 'xkcd:deep sea blue', 'xkcd:grey purple', 'xkcd:dandelion'
    markers = ['o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X']
    labelnames = []
    for x in set(labels):
        labelnames.append(legend[x])
    if labelid == None:
        X_2d = tsne.fit_transform(tfid)
        for i, c, label in zip(target_ids, colors, labelnames):
            ax.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c = c, label = label, marker=markers[i], alpha=0.7, picker=True)
    else: #Implementing the TSNEsingle function here
        clr = colors[labelid]
        mrk = markers[labelid]
        df = pd.DataFrame(tfid)
        df['labels'] = labels
        dfsingle = df.loc[df['labels'] == labelid]
        singlelabel = dfsingle['labels']
        dfsingle = dfsingle.drop(['labels'], axis = 1)
        X_2d = tsne.fit_transform(dfsingle)
        ax.scatter(X_2d[singlelabel == labelid, 0], X_2d[singlelabel == labelid, 1], c = clr, label = legend[labelid], marker = mrk, alpha = 0.7, picker=True)
    plt.title('TSNE(number of datapoints = {})'.format(len(tfid)))
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.86)
    fig.canvas.mpl_connect('pick_event', onclick)
    plt.show()

def visualizeTree(classifier, features, labels, legend, path, name):
    tree = classifier.estimators_[0]
    label = []
    for x in labels:
        label.append(legend[x])
    #Remove 'value' line from the plot
    f = export_graphviz(tree, feature_names=features, class_names=label, precision=2, filled=True, rotate=True)
    f = re.sub(r'\\nvalue = \[.*?\]','',f, flags=re.I)
    outfile = open(path + '\\{}tree.dot'.format(name), 'w')
    outfile.write(f)
    # call([' m', '-Tpng', path + '\\{}tree.dot'.format(name), '-o', path + '\\{}tree.png'.format(name)])

def reviewDoc(y_pred, y_test, doc_test, legend, path=None, name=None):
    correct_pred = y_pred == y_test
    wrong_doc = [] # for wrong predictions
    expected = [] # for wrong predictions
    predicted = [] # for wrong predictions
    correct_doc = [] # for correct predictions
    cor_pred = [] # for correct predictions
    for i, j in zip(correct_pred, range(len(correct_pred))):
        if i:
            correct_doc.append(doc_test[j])
            cor_pred.append(legend[y_test[j]])
        else:
            #print('Expected: {} --> Predicted: {}'.format(legend[y_test[j]], legend[y_pred[j]]))
            wrong_doc.append(doc_test[j])
            expected.append(legend[y_test[j]])
            predicted.append(legend[y_pred[j]])
    # Construct 'wrong' dataframe
    d_wrong = {'Expected': expected, 'Predicted': predicted, 'Document': wrong_doc}
    df_wrong = pd.DataFrame(d_wrong)
    df_wrong = df_wrong.sort_values(by=['Expected'])
    # Construct 'correct' dataframe
    d_correct = {'Label': cor_pred, 'Document':correct_doc}
    df_correct = pd.DataFrame(d_correct)
    df_correct = df_correct.groupby('Label').apply(lambda x: x.sample(5, replace=True)).reset_index(drop=True)
    # Write to xlsx with mutliple sheets
    with pd.ExcelWriter(path + '\\{}Documents.xlsx'.format(name), engine='xlsxwriter') as writer:
        df_wrong.to_excel(writer, sheet_name='Incorrect Docs')
        df_correct.to_excel(writer, sheet_name='Correct Docs')

#def explainText(classifier, vectorizer, labels, legend):
 #   pipe = make_pipeline(vectorizer, classifier)
  #  explainer = LimeTextExplainer(class_names=legend)

def visualizeDecisionBoundary(X, y, classifier, tfid, accscore, method, save = False):
    plt.close()
    plt.cla()
    plt.figure(figsize=(16,9))
    colors = 'r,g,b,c,m,y,k,xkcd:violet blue,orange,purple,xkcd:rose pink,xkcd:aqua green,xkcd:deep sea blue,xkcd:grey purple,xkcd:dandelion' 
    markers = ['o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X','o', '^', 's', 'D', 'X']
    plot_decision_regions(X=X, y=y, clf=classifier, legend=0, colors = colors, markers = markers)
    legend_elements = [
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor ='r', marker = 'o', label = 'Case Management'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'g', marker = '^', label = 'Consult'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'b', marker = 's', label = 'Discharge summary'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'c', marker = 'D', label = 'ECG'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'm', marker = 'X', label = 'Echo'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'y', marker = 'o', label = 'General'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'k', marker = '^', label = 'Nursing'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'xkcd:violet blue', marker = 's', label = 'Nursing_other'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'orange', marker = 'D', label = 'Nutrition'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'purple', marker = 'X', label = 'Pharmacy'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'xkcd:rose pink', marker = 'o', label = 'Physician'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'xkcd:aqua green', marker = '^', label = 'Radiology'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'xkcd:deep sea blue', marker = 's', label = 'Rehab Services'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'xkcd:grey purple', marker = 'D', label = 'Respiratory'),
        Line2D([0],[0], color = 'w', markersize = 10, markerfacecolor = 'xkcd:dandelion', marker = 'X', label = 'Social Work'),
        ]
    plt.legend(handles = legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.title('Accuracy score = {0} / Datapoints = {1} / Classifier = {2}'.format(round(accscore,4), len(X), method))
    if save:
        plt.subplots_adjust(right=0.86)
        plt.savefig('G:\\Casper\\GitKraken\\srp\\Saves\\Results\\DecisionBoundary\\{0}_{1}_{2}.png'.format(method,round(accscore,4),len(X))) 
    else:
        plt.subplots_adjust(right=0.86)
        plt.show()