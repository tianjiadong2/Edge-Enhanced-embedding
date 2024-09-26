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
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

idx_features_labels = np.genfromtxt("./data/cora/cora.content", dtype=np.dtype(str))
edges_unordered = np.genfromtxt("./data/cora/cora.cites", dtype=np.int32)

idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
labels = idx_features_labels[:, -1]


cosine = np.zeros((len(features), len(features)))
for i in range(len(features)):
    print(i)
    for j in range(len(features)):
        cosine[i][j] = np.dot(features[i], features[j]) / (norm(features[i]) * norm(features[j]))

print(cosine)

f = open('Similarity', 'wb')
pickle.dump(cosine, f)
f.close()