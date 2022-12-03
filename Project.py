import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

def main():
    #Read file and save data in an array
    fname = 'bank_note_data.txt'

    with open(fname, 'r') as f:
        data = np.loadtxt(fname, delimiter=',')

    #Shuffle the rows
    np.random.shuffle(data)

    # print(len(data[0]))

    #Find total number of points, number of training data points and number of test data points
    totDataPts = len(data)
    train = round(0.75 * (totDataPts))
    # test = totDataPts - train

    #Split data into test and training data
    trainingData = data[:train, :]
    testData = data[train:, :]

    #Number of attributes
    cols = len(data[0]) - 1

    #Split columns into attributes and class
    train_att, train_class = splitCols(trainingData, cols)
    test_att, test_class = splitCols(testData, cols)

    #Train the classifier
    clf = lda()
    clf.fit(train_att, train_class)
    
    #Test the classifier
    
    # print(test_class[160], '\n')
    # print(clf.predict([test_att[160]]))    
    # print(clf.predict([[-1.99,-6.6,4.82,-0.42]]))

#Splits columns
def splitCols(data, cols):
    x = data[:,:cols]
    y = data[:,-1]
    return x, y

main()