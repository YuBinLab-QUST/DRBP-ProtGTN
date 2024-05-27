import numpy as np
import pandas as pd
import math
import os, sys

binBox = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def readCode(file):
    encodings = []
    if not os.path.exists(file):
        print('Error: file does not exist.')
        sys.exit(1)
    with open(file) as f:
        records = f.readlines()
    for i in records:
        array = i.rstrip().split() if i.strip() != '' else None
        encodings.append(array)
    return np.array(encodings)


def calProb(array):
    myProb = {}
    myClass = set(array)
    for i in myClass:
        myProb[i] = array.count(i) / len(array)
    return myProb


def jointProb(newArray, labels):
    myJointProb = {}
    for i in range(len(labels)):
        myJointProb[str(newArray[i]) + '-' + str(labels[i])] = myJointProb.get(str(newArray[i]) + '-' + str(labels[i]),
                                                                               0) + 1

    for key in myJointProb:
        myJointProb[key] = myJointProb[key] / len(labels)
    return myJointProb


def IG(encodings, num, dim):
    ##########################3

    features = [float(i) for i in range(dim)]
    encodings = np.array(encodings)[:]
    data = encodings[:, :]
    shape = data.shape
    print(shape)
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)

    label1 = np.ones((int(2616), 1))  # Value can be changed
    label2 = np.zeros((int(4175), 1))
    # label1 = np.ones((int(7129), 1))  # Value can be changed
    # label2 = np.zeros((int(7060), 1))
    label_ = np.vstack((label1, label2))
    # labels= label_.tolist()
    labels = [float(i) for i in label_]
    dataShape = data.shape

    if dataShape[1] != len(features):
        print('Error: inconsistent data shape with feature number.')
        return 0, 'Error: inconsistent data shape with feature number.'

    if dataShape[0] != len(labels):
        print('Error: inconsistent data shape with sample number.')
        return 0, 'Error: inconsistent data shape with sample number.'

    probY = calProb(labels)

    myFea = {}
    for i in range(len(features)):
        array = data[:, i]
        newArray = list(pd.cut(array, len(binBox), labels=binBox))

        probX = calProb(newArray)
        probXY = jointProb(newArray, labels)
        HX = -1 * sum([p * math.log(p, 2) for p in probX.values()])
        HXY = 0
        for y in probY.keys():
            for x in probX.keys():
                if str(x) + '-' + str(y) in probXY:
                    HXY = HXY + (probXY[str(x) + '-' + str(y)] * math.log(probXY[str(x) + '-' + str(y)] / probY[y], 2))
        myFea[features[i]] = HX + HXY

    res = []
    ress = []
    res.append(['feature', 'IG-value'])
    for key in sorted(myFea.items(), key=lambda item: item[1], reverse=True):
        res.append([key[0], '{0:.3f}'.format(myFea[key[0]])])
        ress.append([key[0]])
    return res, ress


encodings = pd.read_csv(r'D:\PycharmProjects\my\feature extraction\RBPhunhe2.csv')
data = np.array(encodings)
data = data[:, :]
# print(data.shape)
[m1, n1] = np.shape(data)
num = int(m1 / 2)
# print(num)
dim = int(n1)
# print(dim)
feature, fea = IG(encodings, num, dim)
# print(fea)
feature_index = np.array(fea)
# print(feature_index)
A = [int(x) for x in feature_index]
# A = [int(x) for x in feature_index.ravel()]
set_end = data[:, A[:100]]
data_csv = pd.DataFrame(data=set_end)
data_csv.to_csv('RBP_IG.csv')