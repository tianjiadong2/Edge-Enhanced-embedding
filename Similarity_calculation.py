# -*- coding: utf-8 -*-
# @Author  : Jiadong Tian
# @File    : Similarity_calculation.py
# @Software: PyCharm
# @Instruction: Calculate node similarity

import pickle
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore")


dataset = Planetoid(root='dataset', name='Cora', transform=NormalizeFeatures())
data = dataset[0]
print(data)
x = data.x.detach().numpy()
print(data.x.detach().numpy())

cosine = np.zeros((len(x), len(x)))
for i in range(len(x)):
    print(i)
    for j in range(len(x)):
        cosine[i][j] = np.dot(x[i], x[j]) / (norm(x[i]) * norm(x[j]))

print(cosine)

f = open('Similarity', 'wb')
pickle.dump(cosine, f)
f.close()