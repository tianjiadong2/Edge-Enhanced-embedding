# -*- coding: utf-8 -*-
# @Author  : Jiadong Tian
# @File    : Generation_classification.py
# @Software: PyCharm
# @Instruction: Data generation and classification

import random
import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparameter
K = 40

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
for id in range(int(K * 5429 / 100)):
    adj_add[0].append(adj_list[id][0])
    adj_add[1].append(adj_list[id][1])

# 拼接邻接矩阵
if len(adj_add[0]) > 0 :
    adj_add = torch.tensor(adj_add).to(DEVICE)
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


model_gat = torch.load('./optimum.pth').to(DEVICE)
model_gat.eval()
print(model_gat)
out = model_gat(features, adj_edges)

X = out.detach().cpu().numpy()
y = labels.cpu().numpy()
X_train = out[train_mask].detach().cpu().numpy()
X_test = out[test_mask].detach().cpu().numpy()
y_train = labels[train_mask].cpu().numpy()
y_test = labels[test_mask].cpu().numpy()


process = pd.DataFrame(X_train,columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
process['target'] = y_train

X_for_generate = process.query("target == 1").iloc[:,:-1].values
X_non_default = process.query('target == 0').iloc[:,:-1].values
X_for_generate = torch.tensor(X_for_generate).type(torch.FloatTensor)

# 少数类中的训练集 X
# print(X_for_generate.size())

n_generate = X_non_default.shape[0] - X_for_generate.shape[0]

# 超参数
BATCH_SIZE = 50
LR_G = 0.0001  # G生成器的学习率
LR_D = 0.0001  # D判别器的学习率
N_IDEAS = 20  # G生成器的初始想法(随机灵感)

# 搭建G生成器
G = nn.Sequential(  # 生成器
    nn.Linear(N_IDEAS, 128),  # 生成器等的随机想法
    nn.ReLU(),
    nn.Linear(128, 8),
)

# 搭建D判别器
D = nn.Sequential(  # 判别器
    nn.Linear(8, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),  # 转换为0-1
)

# 定义判别器和生成器的优化器
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

D_loss_ = []

# GAN训练
for step in range(10000):
    for i in range(10):
        # 随机选取BATCH个真实的标签为1的样本
        chosen_data = np.random.choice((X_for_generate.shape[0]), size=(BATCH_SIZE), replace=False)
        artist_paintings = X_for_generate[chosen_data, :]
        # 使用生成器生成虚假样本
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
        G_paintings = G(G_ideas)
        # 使用判别器得到判断的概率
        prob_artist1 = D(G_paintings)
        # 生成器损失
        G_loss = torch.mean(torch.log(1. - prob_artist1))
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

    prob_artist0 = D(artist_paintings)
    prob_artist1 = D(G_paintings.detach())
    # 判别器的损失
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()

    D_loss_.append(D_loss.item())
    D_loss_array = np.array(D_loss_)
    D_loss_sum = sum(abs(D_loss_array[::-1][:20]))
    if D_loss_sum < 0.00002:
        break

fake_data = G(torch.randn(n_generate,N_IDEAS)).detach().numpy()

print("fenleikaishi")
for times in range(10):

    pd.set_option('display.max_columns',5)
    pd.set_option('display.max_rows',4000)

    # (times/100)
    X_default = pd.DataFrame(np.concatenate([X_for_generate,fake_data[:int(1*len(fake_data))]]),columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
    X_default['target'] = 1
    X_non_default = pd.DataFrame(X_non_default,columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
    X_non_default['target'] = 0
    train_data_gan = pd.concat([X_default,X_non_default])

    X_gan = torch.tensor(train_data_gan.iloc[:,:-1].values).type(torch.FloatTensor)
    y_gan = torch.tensor(train_data_gan.iloc[:,-1].values).type(torch.LongTensor)

    X_test = torch.tensor(X_test).type(torch.FloatTensor)
    y_test = torch.tensor(y_test).type(torch.FloatTensor)

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            # 两层感知机
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.drop = torch.nn.Dropout(0.5)
            self.predict = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x

    net = Net(8, 8, 2)  # 输入节点8个，隐层节点8个，输出节点2个

    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)  # 优化器

    def train():
        net.train()
        optimizer.zero_grad()
        prediction = net(X_gan)
        loss = loss_func(prediction, y_gan)
        loss.backward()
        optimizer.step()
        return loss

    def test():
        net.eval()
        out = net(X_test)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        result_pred = pred
        result_true = y_test

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


    best_precision_scor = best_recall_scor = best_auc_scor = best_balanced_accurar = 0
    best_tp = best_fn = best_fp = best_tn = 0

    for epoch in range(1, 100):
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

    log = 'times: {:03d}, best_tp: {:d}, best_fn: {:d}, best_fp: {:d}, best_tn: {:d}, best_recall_score: {:.4f}, best_precision_score: {:.4f}, best_auc_score: {:.4f}'
    print(log.format(times, best_tp, best_fn, best_fp, best_tn, best_recall_scor, best_precision_scor, best_auc_scor))