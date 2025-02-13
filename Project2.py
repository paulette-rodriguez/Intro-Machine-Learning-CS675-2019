from random import *
import sys
from sklearn.svm import LinearSVC

datafile = sys.argv[1]
f = open(datafile)
data = []
i = 0
l = f.readline()

###############
## Read Data ##
###############
while(l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
        data.append(l2)
        l = f.readline()

rows = len(data)
cols = len(data[0])
f.close()

def read_file(filename):
    with open(filename) as datafile:
        data = [line.split() for line in datafile]
        return data

#################
## Read labels ##
#################
labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)

l = f.readline()
while(l != ''):
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1

## Create a random hyperplane ##
def hyperplane(cols):
    w = []
    for j in range(cols-1):
        w.append(0.02 * random() - 0.01)
    w.append(0)
    return w

def sign(x): return 1 if x >= 0 else -1

## Create Projection ##
def projection(data,w):
    rows = len(data)
    cols = len(data[0])
    z = []
    for i in range(rows):
        dp = 0
        for j in range(cols):
            dp += w[j] * float(data[i][j])
        z.append(sign(dp))
    return z

## Running the SVC Classifier on the data ##
def svc_classifier(data,labels,test_data,test_labels):
    classifiers = LinearSVC(max_iter=10000, tol=0.000001).fit(data, labels)
    predicted = classifiers.predict((test_data))
    for i in range(0, len(test_data), 1):
        print(predicted[i] , test_labels[i])


def prj_main():
    data = read_file(sys.argv[1])
    data_labels = read_file(sys.argv[2])
    labels = {int(x[1]): int(x[0]) for x in data_labels}

    matrix = []
    for i in range(2000):
        w = hyperplane(len(data[0]))
        z = projection(data,w)
        matrix.append(z)
    zmatrix = [list(x) for x in zip(*matrix)]
    trainlabels = []
    traindata = []
    testdata =[]
    testlabels = []
    for i in range(len(z)):
        if (labels.get(i) == None):
            testdata.append(zmatrix[i])
            testlabels.append(i)
        else:
            traindata.append(zmatrix[i])
            trainlabels.append(labels.get(i))
    svc_classifier(traindata,trainlabels,testdata,testlabels)

prj_main()
