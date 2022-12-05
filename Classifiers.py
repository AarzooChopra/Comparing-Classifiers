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

    #Shuffle the rows
    np.random.shuffle(data)

    #Find total number of points, number of training data points and number of test data points
    totDataPts = len(data)
    train = round(0.75 * (totDataPts))

    #Split data into test and training data
    trainingData = data[:train, :]
    testData = data[train:, :]

    #Number of attributes
    cols = len(data[0]) - 1

    #Split columns into attributes and class
    train_att, train_class = splitCols(trainingData, cols)
    test_att, test_class = splitCols(testData, cols)

    # x = "-1"
    
    # #Ask user for LDA or SVM
    # while x not in ["0", "1"]:
    #     x = input('\nType "0" for Linear Disciminant Analysis or "1" for Support Vector Machine: ')

    #Train the classifier
    # if x == "0":
    #     clf = lda()
    #     print('\nLinear Discriminant Analysis')
    # else:
    #     clf = svm.SVC(kernel='linear')
    #     print('\nSupport Vector Machine')

    # cm = np.array([0,0,0,0])        #[fp, tp, tn, fn]
    # testLen = len(test_class)

    clf = lda()
    train_time = train_clf(clf, train_att, train_class)
    predict_class, test_time = test(clf, test_att)
    cm = conf_mat(test_class, predict_class)
    print('\nLinear Discriminant Analysis')
    print_stuff(cm, train_time, test_time)

    clf = svm.SVC(kernel='linear')
    train_time = train_clf(clf, train_att, train_class)
    predict_class, test_time =  test(clf, test_att)
    cm = conf_mat(test_class, predict_class)
    print('\nSupport Vector Machine')
    print_stuff(cm, train_time, test_time)

    # train_start = timer()

    # clf.fit(train_att, train_class)
    
    # train_end = timer()

    # test_start = timer()

    # #Test the classifier
    # predict_class = clf.predict(test_att)

    # test_end = timer()
    
    #Build the Confusion Matrix
    # for i in range(testLen):
    #     if test_class[i] == 1:
    #         if predict_class[i] == 0:
    #             cm[0] += 1
    #         else:
    #             cm[2] += 1
    #     else:
    #         if predict_class[i] == 0:
    #             cm[1] += 1
    #         else:
    #             cm[3] += 1

    #Print the Confusion Matrix in a table
    # table = pt(['False Positive: ' + str(cm[0]), 'True Positive: ' + str(cm[1])])
    # table.add_row(['True Negative: ' + str(cm[2]), 'False Negative: ' + str(cm[3])])

    # print('\n' + str(table) + '\n') 
    # print('Training time = ' + str(format((train_end - train_start) * pow(10, 3), '0.2f')) + ' ms\n')  
    # print('Testing time = ' + str(format((test_end - test_start) * pow(10, 3), '0.2f')) + ' ms\n')  
    
    # #Check
    # check(test_class, len(test_class))

#Splits columns
def splitCols(data, cols):
    x = data[:,:cols]
    y = data[:,-1]
    return x, y

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

def train_clf(clf, train_att, train_class):
    train_start = timer()
    clf.fit(train_att, train_class)
    train_end = timer()

    return train_end - train_start

def test(clf, test_att):
    test_start = timer()
    predict_class = clf.predict(test_att)
    test_end = timer()

    return predict_class, test_end - test_start

def conf_mat(test_class, predict_class):
    cm = np.array([0,0,0,0])        #[fp, tp, tn, fn]
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

def print_stuff(cm, train_time, test_time):
    table = pt(['False Positive: ' + str(cm[0]), 'True Positive: ' + str(cm[1])])
    table.add_row(['True Negative: ' + str(cm[2]), 'False Negative: ' + str(cm[3])])

    print('\n' + str(table) + '\n') 
    print('Training time = ' + str(format((train_time) * pow(10, 3), '0.2f')) + ' ms\n')  
    print('Testing time = ' + str(format((test_time) * pow(10, 3), '0.2f')) + ' ms\n')  


main()
