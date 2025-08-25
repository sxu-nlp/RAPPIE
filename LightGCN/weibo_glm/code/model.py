"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import pickle
import numpy as np


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        # self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # self.embedding_item = torch.nn.Embedding(
        #     num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.linear_adapter1 = nn.Linear(4096, 2048)
        self.linear_adapter2 = nn.Linear(9, 2048)
        self.linear_adapter3 = nn.Linear(4096, 4096)
        self.linear_adapter4 = nn.Linear(4096, 2048)
        self.linear_adapter5 = nn.Linear(4096, 2048)
        self.linear_adapter6 = nn.Linear(4096, 4096)
        self.linear_adapter7 = nn.Linear(4096, 2048)
        self.linear_adapter8 = nn.Linear(4096, 4096)
        self.linear_adapter9 = nn.Linear(4096, 4096)
        self.multi_head_attention1 = torch.nn.MultiheadAttention(4096, 8, dropout=0)
        self.multi_head_attention2 = torch.nn.MultiheadAttention(4096, 8, dropout=0)

        user_in_emb_path = "../data/" + world.dataset + "/user_in_features.pkl"
        user_a_emb_path = "../data/" + world.dataset + "/user_attitude.pkl"
        role_emb_path = "../data/" + world.dataset + "/role_features.pkl"
        user_in_emb = self.readEmbedding(user_in_emb_path)
        user_a_emb = self.readEmbedding(user_a_emb_path)
        role_emb = self.readEmbedding(role_emb_path)
        user_in_emb_tensor = torch.tensor(user_in_emb, dtype=torch.float)
        user_a_emb_tensor = torch.tensor(user_a_emb, dtype=torch.float)
        role_emb_tensor = torch.tensor(role_emb, dtype=torch.float)
        attention_outputs = []
        for i in range(3508):
            a = self.linear_adapter8(user_in_emb_tensor[i]).unsqueeze(0).unsqueeze(1)
            b = self.linear_adapter9(user_a_emb_tensor[i]).unsqueeze(0).unsqueeze(1)
            attn_output = self.multi_head_attention1(a, b, b)[0]
            attention_outputs.append(attn_output.squeeze(1).squeeze(0))
        user_embedding_tensor_0 = torch.stack(attention_outputs)
        user_embedding_tensor_1 = self.linear_adapter3(torch.cat((self.linear_adapter1(user_in_emb_tensor), self.linear_adapter7(user_embedding_tensor_0)), dim=1))
        user_embedding_tensor_2 = self.multi_head_attention2(user_embedding_tensor_1, role_emb_tensor, role_emb_tensor)[0]
        user_embedding_tensor = self.linear_adapter6(torch.cat((self.linear_adapter4(user_embedding_tensor_1), self.linear_adapter5(user_embedding_tensor_2)), dim=1))

        # 将加载的 Tensor 分配给 embedding_user.weight.data
        self.embedding_user.weight.data = user_embedding_tensor
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def readEmbedding(self, user_file):
        '''
        load the pretrain svd embedding
        '''
        with open(user_file, 'rb') as file1:
            user_embedding = pickle.load(file1)
        #with open(item_file, 'rb') as file2:
        #    item_embedding = pickle.load(file2)
        #return user_embedding, item_embedding
        return user_embedding

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        # all_emb = torch.cat([users_emb, items_emb])
        all_emb = users_emb
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users = light_out
        # users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users#, items
    
    def getUsersRating(self, users):
        all_users = self.computer()
        # all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        # items_emb = all_items
        rating = self.f(torch.matmul(users_emb, all_users.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users = self.computer()
        users_emb = all_users[users]
        pos_emb = all_users[pos_items]
        neg_emb = all_users[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_user(pos_items)
        neg_emb_ego = self.embedding_user(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

    def saveEmbedding(self,filename):
        with open(filename,'wb') as file:
            pickle.dump(self.embedding_user.weight.data,file)
        print("write over!")
