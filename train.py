import os.path as osp
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *
from sklearn import preprocessing
# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim
from sklearn.metrics import accuracy_score
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder_imp,Explainer,Encoder
from evaluate_embedding import evaluate_embedding
from model import *
from model import propty_attack

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb
import os
import scipy as sp

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC


def convert_to_one_hot(Y,C):
    Y=np.eye(C)[Y.reshape(-1)]
    return Y
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()









class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder_imp(dataset_num_features, hidden_dim, num_gc_layers,pooling='all')
    # self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
    self.explainer = Explainer(dataset_num_features, hidden_dim, 3)

    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, node_imp):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch,node_imp)
    
    y = self.proj_head(y)
    
    return y

  def explain(self, x, edge_index, batch):

      if x is None:
          x = torch.ones(batch.shape[0]).to(device)

      y = self.explainer(x, edge_index, batch)

      return y

  def loss_cal(self, x, x_aug):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss


import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def tensor_edge_to_numpy_adj(edge_index):
    node_num=torch.max(edge_index).item()
    # print(node_num)
    adj=np.zeros((node_num+1,node_num+1))
    for i in range(edge_index.size()[1]):
        adj[edge_index[0][i]][edge_index[1][i]]=1
    return adj


def cal_density(adj):
    num_node=adj.shape[0]
    num_edge=np.count_nonzero(adj)/2
    return num_edge/(num_node*num_node)

def cal_node(adj):
    num_node=adj.shape[0]

    return num_node

def cal_degree(adj):
    num_node=adj.shape[0]
    num_edge = np.count_nonzero(adj)

    return num_edge/(num_node)

def _generate_bin(attr, num_class):
    sort_attr = np.sort(attr)
    bins = np.zeros(num_class - 1)
    unit = attr.size / num_class
    for i in range(num_class - 1):
        bins[i] = (sort_attr[int(np.floor(unit * (i + 1)))] + sort_attr[int(np.ceil(unit * (i + 1)))]) / 2

    return bins


def label_smoothing(result):
    noise=np.ones_like(result)
    noise=0.5*noise
    result=0.2*result+0.8*noise
    return result







def logloss(y_true, y_pred):
    y_true_0 = y_true[0]
    y_true_1 = y_true[1]
    score_0 = y_true_0 == y_pred
    score_1 = y_true_1 != y_pred
    score_0_norm = np.average(score_0)
    score_1_norm = np.average(score_1)
    score=0.9*score_0_norm+0.1*score_1_norm


    return score


# 这里的greater_is_better参数决定了自定义的评价指标是越大越好还是越小越好
# loss = make_scorer(logloss, greater_is_better=False)
# score = make_scorer(accuracy_score, greater_is_better=True)

def DP_noise(epsilon,shape,mode='gauss'):
    delta=10e-5
    if mode=='gauss':
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / epsilon
        noise=np.random.normal(loc=0,scale=sigma,size=shape)
    elif mode=='laplace':
        noise = np.random.laplace(loc=0,scale=1/epsilon,size=shape)
    else:
        print('do not find {} model'.format(mode))
    # print(noise.dtype)
    noise=noise.astype(np.float32)
    # print(noise.dtype)
    return torch.from_numpy(noise)



if __name__ == '__main__':
    
    args = arg_parse()
    setup_seed(args.seed)

    accuracies_1 = {'val':[], 'test':[]}
    epochs = 20
    log_interval = 10
    batch_size = 1
    num_class=2
    # batch_size = 512
    lr = args.lr
    DS = args.DS
    # DS = 'MUTAG'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    # dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset = TUDataset(path, name=DS, aug=args.aug)
    # dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none')
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size,shuffle=False)

    train_propty = []
    test_propty = []


    for data in dataloader:
        data,_=data
        edge_index=data.edge_index
        adj=tensor_edge_to_numpy_adj(edge_index)
        density=cal_density(adj)
        train_propty.append(density)

    for data in dataloader_eval:
        data,_=data
        edge_index=data.edge_index
        adj=tensor_edge_to_numpy_adj(edge_index)
        density=cal_density(adj)
        test_propty.append(density)

    train_propty=np.array(train_propty)
    test_propty = np.array(test_propty)

    # print(train_propty)

    bins = _generate_bin(train_propty, num_class)
    train_label = np.digitize(train_propty, bins)
    test_label = np.digitize(test_propty, bins)
    # print(train_label)



    # sys.exit()

    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    model.eval()
    # emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """


    batch_size = 512

    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size,shuffle=False)
    SHIJIAN_1=time.time()

    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()
        for data in dataloader:

            # print('start')
            data, data_aug = data
            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device)

            data_imp = model.explain(data.x, data.edge_index, data.batch)
            x = model(data.x, data.edge_index, data.batch, data_imp)

            # x = model(data.x, data.edge_index, data.batch, data.num_graphs)
            # print(x.size())
            # sys.exit()

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                # node_num_aug, _ = data_aug.x.size()
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                            not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)

            '''
            print(data.edge_index)
            print(data.edge_index.size())
            print(data_aug.edge_index)
            print(data_aug.edge_index.size())
            print(data.x.size())
            print(data_aug.x.size())
            print(data.batch.size())
            print(data_aug.batch.size())
            pdb.set_trace()
            '''
            data_aug_imp = model.explain(data_aug.x, data_aug.edge_index, data_aug.batch)
            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug_imp)

            # print(x)
            # print(x_aug)
            loss = model.loss_cal(x, x_aug)
            print(loss)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            # print('batch')
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            # emb, y = model.encoder.get_embeddings(dataloader_eval)
            # print()
            #
            # acc_val, acc = evaluate_embedding(emb, y)
            # accuracies['val'].append(acc_val)
            # accuracies['test'].append(acc)
            # print(accuracies['val'][-1], accuracies['test'][-1])

    SHIJIAN_2 = time.time()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_save_path = os.path.join(current_dir, '../PaGCL/model/gsimclr_{}_{}_DEFENSE.pth'.format(args.DS, args.aug))
    torch.save(model.state_dict(), model_save_path)
    # emb_train, y_train = model.encoder.get_embeddings(dataloader)
    emb_test, y_test = model.encoder.get_embeddings(dataloader_eval)

    noise=DP_noise(1,emb_test.shape)

    emb_test=noise+emb_test






    acc_val, acc = evaluate_embedding(emb_test, y_test)
    accuracies_1['val'].append(acc_val)
    accuracies_1['test'].append(acc)
    print(accuracies_1['val'][-1], accuracies_1['test'][-1])
    # sys.exit()





    ######################################################

    labels = preprocessing.LabelEncoder().fit_transform(y_test)
    x, y = np.array(emb_test), np.array(labels)

    y_all=np.vstack((y,test_label))

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies=[]
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        # classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier = GridSearchCV(SVC(), params, cv=5, scoring=make_scorer(accuracy_score, greater_is_better=True), verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))


    pred_decision=classifier.decision_function(x)
    pred_decision=pred_decision.reshape((pred_decision.shape[0],-1))
    pred = classifier.predict(x)
    print(pred.shape)
    # sys.exit()
    pred_onehot=convert_to_one_hot(pred,(pred.max()+1))
    # pred_onehot=np.hstack((pred_onehot,pred_decision))

    # pred_onehot=label_smoothing(pred_onehot)



    x_sig=sigmoid(x)



    emb_train=torch.from_numpy(pred_onehot[:int(len(pred_onehot)/2)])
    emb_train=emb_train.type(torch.FloatTensor)
    emb_test=torch.from_numpy(pred_onehot[int(len(pred_onehot)/2):])
    emb_test=emb_test.type(torch.FloatTensor)

    train_label=torch.from_numpy(train_label[:int(len(pred_onehot)/2)])
    train_label=train_label.type(torch.LongTensor)
    test_label=torch.from_numpy(test_label[int(len(pred_onehot)/2):])
    test_label=test_label.type(torch.LongTensor)

    train_data_label = torch.utils.data.TensorDataset(emb_train, train_label)
    test_data_label = torch.utils.data.TensorDataset(emb_test, test_label)

    train_loader = torch.utils.data.DataLoader(dataset=train_data_label, batch_size=128,
                                               shuffle=True, num_workers=1, drop_last=False)
    validate_loader = torch.utils.data.DataLoader(dataset=test_data_label, batch_size=128, shuffle=False,
                                                  num_workers=1, drop_last=False)




    attack_model=propty_attack(pred_onehot.shape[1],num_class)
    opt_attack=torch.optim.Adam(attack_model.parameters(), lr=lr)

    best_acc = 100
    std = []

    for epoch in range(epochs):
        loss_all=0.0
        predict_label_all = []
        ori_label_all = []
        for index,(data,labels) in enumerate(train_loader):

            attack_model.train()
            opt_attack.zero_grad()
            predict=attack_model(data)
            predict_att=torch.max(predict,1)[1]

            loss = F.cross_entropy(predict, labels)
            loss_all+=loss.item()
            loss.backward()

            opt_attack.step()
            attack_model.eval()

            predict_att = predict_att.cpu().numpy()

            labels = labels.cpu().numpy()
            predict_label_all.extend(predict_att)
            ori_label_all.extend(labels)
        predict_label_all = np.array(predict_label_all)
        ori_label_all = np.array(ori_label_all)

        acc = 0
        for i in range(len(predict_label_all)):
            if predict_label_all[i] == ori_label_all[i]:
                acc += 1

        print('train_acc:', acc / len(predict_label_all))




        print('epoch:', epoch, 'loss_dis:', loss_all / (index + 1))



        predict_label_all = []
        ori_label_all = []
        for index, (data, labels) in enumerate(validate_loader):
            attack_model.eval()
            predict_logit = attack_model(data)
            predict_label = torch.max(predict_logit, 1)[1]
            predict_label = predict_label.cpu().numpy()

            labels = labels.cpu().numpy()
            predict_label_all.extend(predict_label)
            ori_label_all.extend(labels)
        predict_label_all = np.array(predict_label_all)
        ori_label_all = np.array(ori_label_all)
        acc = 0
        for i in range(len(predict_label_all)):
            if predict_label_all[i] == ori_label_all[i]:
                acc += 1
        test_acc=acc / len(predict_label_all)
        print('test_acc:', acc / len(predict_label_all))

        if epoch>=10 and test_acc<=best_acc:
            best_acc=test_acc
        if epoch>=(epochs-20):
            std.append(test_acc)







    std=np.array(std)
    std=np.std(std)
    print('best_acc:',best_acc)
    print('std:',std)
    print(accuracies_1['val'][-1], accuracies_1['test'][-1])
    print('Time:',SHIJIAN_2-SHIJIAN_1)




