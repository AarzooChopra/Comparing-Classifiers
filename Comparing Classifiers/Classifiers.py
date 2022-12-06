"""
Data Analytics and Big Data
Final Project

Comparing Classifiers

This program uses 2 menthods to train and test a classifier: Linear Didcriminant Analysis and Support Vector Machine
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn import svm
from prettytable import PrettyTable as pt
from timeit import default_timer as timer

def main():

    #Read file and save data in an array
    fname = 'bank_note_data.txt'
    with open(fname, 'r') as f:
        data = np.loadtxt(fname, delimiter=',')

    np.random.shuffle(data)             #Shuffle the rows
    cols = len(data[0]) - 1             #Number of attributes

    totDataPts = len(data)              #Total data points
    train = round(0.75 * (totDataPts))  #Number of training data points
    
    #Split data into test and training data
    trainingData = data[:train, :]
    testData = data[train:, :]

    #Split columns into attributes and class
    train_att, train_class = splitCols(trainingData, cols)
    test_att, test_class = splitCols(testData, cols)

    #Linear Discriminant Analysis
    clf = lda()
    print('\nLinear Discriminant Analysis')
    func_call(clf, train_att, train_class, test_att,test_class)

    #Support Vector Machine
    clf = svm.SVC(kernel='linear')
    print('\nSupport Vector Machine')
    func_call(clf, train_att, train_class, test_att,test_class)
    
    # #Check
    # check(test_class, len(test_class))

#Splits columns
def splitCols(data, cols):
    x = data[:,:cols]
    y = data[:,-1]
    
    return x, y

#Train the classifier
def train_clf(clf, train_att, train_class):
    train_start = timer()
    clf.fit(train_att, train_class)
    train_end = timer()

    return train_end - train_start

#Test the classifier
def test_clf(clf, test_att):
    test_start = timer()
    predict_class = clf.predict(test_att)
    test_end = timer()

    return predict_class, test_end - test_start

#Build the Confusion Matrix
def conf_mat(test_class, predict_class):
    cm = np.array([0,0,0,0])            #[fp, tp, tn, fn]
    testLen = len(test_class)

    for i in range(testLen):
        if test_class[i] == 1:
            if predict_class[i] == 0:
                cm[0] += 1
            else:
                cm[2] += 1
        else:
            if predict_class[i] == 0:
                cm[1] += 1
            else:
                cm[3] += 1
    
    return cm

#Print the results
def print_stuff(cm, train_time, test_time):
    #Build table
    table = pt(['False Positive: ' + str(cm[0]), 'True Positive: ' + str(cm[1])])
    table.add_row(['True Negative: ' + str(cm[2]), 'False Negative: ' + str(cm[3])])
    #Print
    print('\n' + str(table) + '\n') 
    print('Training time = ' + str(format((train_time) * pow(10, 3), '0.2f')) + ' ms\n')  
    print('Testing time = ' + str(format((test_time) * pow(10, 3), '0.2f')) + ' ms\n')  

#Function calls
def func_call(clf, train_att, train_class, test_att,test_class):
    train_time = train_clf(clf, train_att, train_class)
    predict_class, test_time = test_clf(clf, test_att)
    cm = conf_mat(test_class, predict_class)
    print_stuff(cm, train_time, test_time)

#Check if the confusion matrix is correct (number of elements in each class)
def check(test_class, testLen):
    a = 0
    b = 0
    for i in range(testLen):
        if test_class[i] == 0:
            a += 1
        else:
            b += 1
    print('Number of elements in A = ', a, '\nNumber of elements in B = ', b, '\n') 

main()
