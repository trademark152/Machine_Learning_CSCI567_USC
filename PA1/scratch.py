import numpy as np
from collections import Counter
import utils

trainY=[1,0,1,1,0]
numTest = 3
labels = np.tile(trainY, (numTest, 1))
print(labels)
for idx, row in (enumerate(labels)):
    print(idx)
    print(row)
    print(np.argsort(row))
    row1 = row[np.argsort(row)]
    print(row1)
    print(Counter(row1[:2]))

features = [[3, 4], [1, -1], [0, 0]]
x=utils.NormalizationScaler()
print(features)
print(x(features))

features = [[2, -1], [-1, 5], [0, 0]]
y=utils.MinMaxScaler()
print(features)
print(y(features))