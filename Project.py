import numpy as np
import pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

    #Read data into array
data = pandas.read_excel("fld1.xlsx", header=None)
data_npy = pandas.DataFrame.to_numpy(data)

X = data_npy[:,:2]
y = data_npy[:,-1]
# print(y)
# X = np.array([data_npy[]])
# y = np.array([1, 1, 1, 2, 2, 2])
clf = lda()
clf.fit(X, y)

print(clf.predict([[1.4, -0.2]]))
