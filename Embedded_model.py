# -*- coding: utf-8 -*-
# @Author  : Jiadong Tian
# @File    : Embedded_model.py
# @Software: PyCharm
# @Instruction: Select and save the optimal embedding model

import random
import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
import numpy as np
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 处理连接
f = open('cosine', 'rb')
prob_adj = pickle.load(f)
f.close()

adj_list = []

if os.path.exists('adj_list'):
    f = open('adj_list', 'rb')
    adj_list = pickle.load(f)
    f.close()
    print(len(adj_list))

else:
    # 提取多少倍连接，10倍
    for ii in range(int(110 * 5429 / 10)):

        if ii % 1000 == 0:
            print(ii,"/",110 * 5429 / 10)
        prob_adj_max = prob_adj.max()  # 最大值
        index = np.unravel_index(prob_adj.argmax(), prob_adj.shape)  # 最大值索引
        prob_adj[index] = -100

        if index[0] == index[1]:
            continue
        if (index in adj_list) or (index[1], index[0]) in adj_list:
            continue
        else:
            adj_list.append(index)

    f = open('adj_list', 'wb')
    pickle.dump(adj_list, f)
    f.close()

# 多少个步
for bufen in range(0, 100):

    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    idx_features_labels = np.genfromtxt("./data/cora/cora.content", dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("./data/cora/cora.cites", dtype=np.int32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]

    classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 0, 'Probabilistic_Methods': 0, 'Case_Based': 0,
                    'Theory': 0, 'Rule_Learning': 1, 'Genetic_Algorithms': 0}
    labels = np.array(list(map(classes_dict.get, labels)))

    idx_dict = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_dict.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    features = torch.FloatTensor(np.array(normalize(features).todense())).to(DEVICE)
    labels = torch.LongTensor(labels).to(DEVICE)

    adj_edges = [[], []]
    for i in edges:
        adj_edges[0].append(i[0])
        adj_edges[1].append(i[1])
    adj_edges = torch.LongTensor(adj_edges).to(DEVICE)

    # 检查随机文件是否存在，存在则读取，不存在则生成并存储
    mask = []
    if os.path.exists('mask'):
        f = open('mask', 'rb')
        mask = pickle.load(f)
        f.close()
    else:
        for i in range(0, 2708):
            random_mask = random.randint(1, 10)
            mask.append(random_mask)
        f = open('mask', 'wb')
        pickle.dump(mask, f)
        f.close()

    train_mask = []
    val_mask = []
    test_mask = []

    for i in mask:
        if i <= 7:
            train_mask.append(True)
            val_mask.append(False)
            test_mask.append(False)
        elif i == 8:
            train_mask.append(False)
            val_mask.append(True)
            test_mask.append(False)
        else:
            train_mask.append(False)
            val_mask.append(False)
            test_mask.append(True)

    train_mask = torch.tensor(train_mask).to(DEVICE)
    val_mask = torch.tensor(val_mask).to(DEVICE)
    test_mask = torch.tensor(test_mask).to(DEVICE)

# 百分比，十分比
    adj_add = [[], []]
    for id in range(int(bufen * 5429 / 100)):
        adj_add[0].append(adj_list[id][0])
        adj_add[1].append(adj_list[id][1])

    # 拼接邻接矩阵
    if len(adj_add[0]) > 0 :
        adj_add = torch.tensor(adj_add)
        print(adj_edges.size())
        adj_edges = torch.cat((adj_edges, adj_add), 1)
        print(adj_edges.size())

    class GCN(torch.nn.Module):
        def __init__(self, feature, hidden, classes):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(feature, hidden)
            self.conv2 = GCNConv(hidden, classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return x


    model = GCN(1433, 256, 8).to(DEVICE)
    # print(model)

    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # 优化器

    best_precision_scor = best_recall_scor = best_auc_scor = best_balanced_accurar = 0
    best_tp = best_fn = best_fp = best_tn = 0

    # 每个工况重复100次
    for cishu in range(100):
        print(cishu)
        def train():
            model.train()
            x, edge_index = features, adj_edges
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            return loss

        def test():
            model.eval()
            out = model(features, adj_edges)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            result_pred = pred[test_mask].cpu().numpy()
            result_true = labels[test_mask].cpu().numpy()

            tp = 0
            fn = 0
            fp = 0
            tn = 0
            for i in range(0, len(result_pred)):
                if result_pred[i] == 1 and result_true[i] == 1:
                    tp = tp + 1
                elif result_pred[i] == 0 and result_true[i] == 1:
                    fn = fn + 1
                elif result_pred[i] == 1 and result_true[i] == 0:
                    fp = fp + 1
                elif result_pred[i] == 0 and result_true[i] == 0:
                    tn = tn + 1

            from sklearn.metrics import precision_score
            precision_scor = precision_score(result_true, result_pred)

            from sklearn.metrics import recall_score
            recall_scor = recall_score(result_true, result_pred)

            from sklearn.metrics import roc_auc_score
            auc_scor = roc_auc_score(result_true, result_pred)

            return tp, fn, fp, tn, precision_scor, recall_scor, auc_scor

        for epoch in range(1, 500):
            train_loss = train()
            tp, fn, fp, tn, precision_scor, recall_scor, auc_scor = test()
            # 以auc为准
            if auc_scor > best_auc_scor:
                best_tp = tp
                best_fn = fn
                best_fp = fp
                best_tn = tn
                best_recall_scor = recall_scor
                best_precision_scor = precision_scor
                best_auc_scor = auc_scor
                torch.save(model, './optimum.pth')
                print("模型已更新")

                log = 'bufen: {:03d}, best_tp: {:.4f}, best_fn: {:.4f}, best_fp: {:.4f}, best_tn: {:.4f}, best_recall_score: {:.4f}, best_precision_score: {:.4f}, best_auc_score: {:.4f}'
                print(log.format(bufen, best_tp, best_fn, best_fp, best_tn, best_recall_scor, best_precision_scor,best_auc_scor,best_balanced_accurar))