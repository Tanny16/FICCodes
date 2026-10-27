
import numpy as np
import csv
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, to_undirected
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
np.set_printoptions(threshold=np.inf)
from training.mytools import *
class Attn_head_adj(nn.Module):  #用于处理行业关系超图中第一层注意力（邻居的重要性）
    def __init__(self,
                 in_channel,
                 out_sz,
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=None,
                 residual=False):
        super(Attn_head_adj, self).__init__()

        self.in_channel = in_channel
        self.out_sz = out_sz
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual

        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout()
        self.coef_dropout = nn.Dropout()
        self.res_conv = nn.Conv1d(self.in_channel, self.out_sz, 1)

    def forward(self, x, adj): #x torch.Size([32, 758, 32])

        #seq = x.permute(0, 2, 1)  #seq [32, 32, 758]
        seq= x.transpose(0, 1).unsqueeze(0) #x [758,32]变成[32,758]变成[1,32,758],因为卷积变换第二个维度
        seq_fts = self.conv1(seq) #seq [1,32,758]变成[1,8,758]
        f_1 = self.conv2_1(seq_fts)  #([1, 8, 758])
        f_2 = self.conv2_2(seq_fts)  #([1, 8, 758])

        logits = f_1 + torch.transpose(f_2, 2, 1) #[1, 758, 758]
        logits = self.leakyrelu(logits) #logits.shape torch.Size([1, 758, 758])
        zero_vec = -9e15 * torch.ones_like(logits)
        attention = torch.where(adj > 0, logits, zero_vec)

        coefs = self.softmax(attention)
        #self.coef_drop=0.6 self.in_dropout Dropout(p=0.5, inplace=False)
        if self.coef_drop != 0.0:  #对新生成的行业关系的attention进行dropout
            coefs = self.coef_dropout(coefs) #coefs [1, 758, 758]
        if self.in_dropout != 0.0:  #对原始输入进行dropout
            seq_fts = self.in_dropout(seq_fts) #torch.Size([1, 8, 758])

        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))  #torch.Size([1, 758, 8])

        if self.residual: #true  是否进行残差连接
            if seq.shape[1] != ret.shape[2]:
                ret = ret + self.res_conv(seq).permute(0, 2, 1) #由于ret [1, 758, 8]  seq [1,32,758]，则seq先permute成[1,758,32]再经过一个卷积才可以维度匹配
            else:
                ret = ret + seq.permute(0, 2, 1)

        return self.activation(ret)  #ELU(alpha=1.0) [1, 758, 8]

class Attn_head(nn.Module):  #用于处理基金关系超图中第一层注意力（邻居的重要性）
    def __init__(self,
                 in_channel,
                 out_sz,
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=None,
                 residual=False):
        super(Attn_head, self).__init__()

        self.in_channel = in_channel
        self.out_sz = out_sz
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual

        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout()
        self.coef_dropout = nn.Dropout()
        self.res_conv = nn.Conv1d(self.in_channel, self.out_sz, 1)

    def forward(self, x): #torch.Size([32, 6, 32]) 其中32是batch 6是当前基金中的一个注意是其中一个关系，32是这6只股票的特征


        seq= x.transpose(0, 1).unsqueeze(0)
        seq_fts = self.conv1(seq) #卷积特征变换从[1, 32, 6] 到 torch.Size([1, 8, 6])

        f_1 = self.conv2_1(seq_fts)  #[1, 1, 6]
        f_2 = self.conv2_2(seq_fts)  #[1, 1, 6]
        logits = f_1 + torch.transpose(f_2, 2, 1)  #[1, 6, 6]
        logits = self.leakyrelu(logits)
        coefs = self.softmax(logits)

        if self.coef_drop != 0.0: #true
            coefs = self.coef_dropout(coefs)
        if self.in_dropout != 0.0: #true
            seq_fts = self.in_dropout(seq_fts)

        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))  #coefs [1, 6, 6]   seq_fts [1, 8, 6]——trans——[1, 6, 8]    [1, 6, 6]matmul[1, 6, 8]变成[1, 6, 8]

        if self.residual: #true 这一步的目的就是加入残差项，需要要求变换后的嵌入与原始输入维度相同，如果不相同的话，就通过卷积变换和trans变换使二者的维度相同
            if seq.shape[1] != ret.shape[2]:
                ret = ret + self.res_conv(seq).permute(0, 2, 1)  #seq本来是[32, 32, 6]经过卷积变成[32, 8, 6]，再经过permute变成[32, 6, 8]，此时与ret相同了
            else:
                ret = ret + seq.permute(0, 2, 1)

        return self.activation(ret) #self.activation ELU(alpha=1.0)      torch.Size([32, 6, 8]) batch 股票数 特征  （此时加入了每个股票的权重）

class edge_attn(nn.Module):  #这一步才是真正处理超边间注意力，对于当前的股票i，哪个超边更重要，输入hyperedge_fts_i为[32, 2, 8]，其中2代表当前股票出现在2个超边

    def __init__(self,
                 in_channel,
                 out_sz):
        super(edge_attn, self).__init__()
        self.in_channel = in_channel
        self.out_sz = out_sz

        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        seq_fts = self.conv1(x.permute(0, 2, 1))
        f_1 = self.conv2_1(seq_fts)
        f_2 = self.conv2_2(seq_fts)
        logits = f_1 + torch.transpose(f_2, 2, 1)
        logits = self.leakyrelu(logits)

        coefs = self.softmax(logits)
        coefs = torch.sum(coefs, dim=-1, out=None)
        coefs = self.softmax(coefs)
        coefs = coefs.unsqueeze(1)
        return coefs



import copy  # 导入 copy 模块

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
class HeteGNN(nn.Module):
    def __init__(self, input_dim, output_dim, rel_num, negative_slope=0.2):
        super(HeteGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rel_num = rel_num
        self.negative_slope = negative_slope

        # 可学习的权重矩阵 W
        self.W = nn.Linear(2 * output_dim, output_dim, bias=False)

        # 共享的注意力向量 a_n
        self.attention_vector = nn.Parameter(torch.Tensor(output_dim, 1))
        nn.init.xavier_uniform_(self.attention_vector)

        # 投影层
        self.proj_com = nn.Linear(input_dim, output_dim, bias=False)  # 投影公司特征
        self.proj_per = nn.Linear(input_dim, output_dim, bias=False)  # 投影高管特征
        self.proj_com2 = nn.Linear(input_dim, output_dim, bias=False)
        self.proj_per2 = nn.Linear(input_dim, output_dim, bias=False)

        # 批量归一化
        self.bn = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        # 激活函数 δ
        self.delta = nn.ReLU()

    def forward(self, x, heter_adj, matrix,rela_list_bi=None):
        # 处理第一个邻接矩阵，表示公司-高管关系
        e_f_matrix = matrix[0].T
        e_f_matrix = e_f_matrix.float()  # 转换为浮点数
        e_f = e_f_matrix @ x  # 高管的特征矩阵
        company_emb = self.proj_com(x)
        exe_emb = self.proj_per(e_f)
        emb = torch.cat((company_emb, exe_emb), dim=0)
        emb = self.bn(emb)
        # 处理第二个邻接矩阵，表示高管-股东关系
        s_f_matrix = matrix[1].T
        s_f_matrix = s_f_matrix.float()  # 转换为浮点数
        s_f = s_f_matrix @ x  # 股东的特征矩阵
        company_emb2 = self.proj_com2(x)
        sha_emb = self.proj_per2(s_f)

        # 拼接特征并归一化
        emb2 = torch.cat((company_emb2, sha_emb), dim=0)
        emb2 = self.bn2(emb2)
        total_emb = []
        heter_adj1 = copy.deepcopy(heter_adj)

        #heter_adj1[1] = heter_adj1[1]-x.shape[0]
        # 节点注意力计算
        total_emb.append(self.compute_attention(heter_adj1[0], company_emb, emb)[:x.shape[0]])  # 对第一种关系应用注意力机制
        #total_emb.append(self.compute_attention(heter_adj[1], emb, emb)[:x.shape[0]])  # 对第二种关系应用注意力机制
        total_emb.append(self.compute_attention( heter_adj1[1], company_emb2, emb2)[:x.shape[0]]) # 对第二种关系应用注意力机制

        # Step 4: 添加公司A-经理1-经理2-公司B的元路径传播
        # 公司A -> 经理1 -> 经理2 -> 公司B
        #company_to_manager_to_manager_to_company = self.metapath_propagation(heter_adj1[0], heter_adj1[1], company_emb,
        #                                                                     exe_emb, emb)
        #total_emb.append(company_to_manager_to_manager_to_company)

        return total_emb,x

    def metapath_propagation(self, heter_adj_0, heter_adj_1, company_emb, exe_emb, emb):
        """
        实现公司A -> 经理1 -> 经理2 -> 公司B 的元路径传播
        heter_adj_0: 公司到经理的邻接矩阵
        heter_adj_1: 经理到经理的邻接矩阵
        company_emb: 公司嵌入
        exe_emb: 高管嵌入
        emb: 总嵌入
        """
        heter_adj_0 = heter_adj_0.clone()
        heter_adj_0[:, 1] = heter_adj_0[:, 1] - company_emb.shape[0]
        # 第一步：从公司A传播信息到经理1
        manager1_emb = self.compute_attention2(heter_adj_0, company_emb, exe_emb)

        # 第二步：从经理1传播信息到经理2
        manager2_emb = self.compute_attention(heter_adj_1, manager1_emb, exe_emb)

        # 第三步：从经理2传播信息到公司B
        heter_adj_0 = heter_adj_0[:, [1, 0]].clone()
        company_updated_emb = self.compute_attention2(heter_adj_0, manager2_emb, company_emb)

        # 返回更新后的公司嵌入
        return company_updated_emb
    def compute_attention(self, adj, h_i, h_j):
        # 源节点和目标节点
        src_nodes = adj[:, 0]
        dst_nodes = adj[:, 1]

        # 拼接源节点和目标节点的特征
        h_ij = torch.cat([h_i[src_nodes], h_j[dst_nodes]], dim=-1)

        # 计算注意力得分
        r_ij = F.leaky_relu(self.W(h_ij) @ self.attention_vector, negative_slope=self.negative_slope)

        # softmax 归一化注意力
        attention_weights = torch.zeros_like(r_ij)
        unique_src = torch.unique(src_nodes)
        for node in unique_src:
            mask = (src_nodes == node)
            attention_weights[mask] = F.softmax(r_ij[mask].clone(), dim=0)

        # 加权聚合邻居特征
        weighted_h_j = attention_weights * h_j[dst_nodes]
        updated_h_i = torch.zeros_like(h_i)
        #updated_h_i.index_add_(0, src_nodes, weighted_h_j)
        updated_h_i = updated_h_i.index_add(0, src_nodes, weighted_h_j)

        # 非线性激活
        updated_h_i = self.delta(updated_h_i)

        return updated_h_i

    def compute_attention2(self, adj, h_i, h_j):
        # 源节点和目标节点
        src_nodes = adj[:, 0]
        dst_nodes = adj[:, 1]

        # 拼接源节点和目标节点的特征
        h_ij = torch.cat([h_i[src_nodes], h_j[dst_nodes]], dim=-1)

        # 计算注意力得分
        r_ij = F.leaky_relu(self.W(h_ij) @ self.attention_vector, negative_slope=self.negative_slope)

        # softmax 归一化注意力
        attention_weights = torch.zeros_like(r_ij)
        unique_src = torch.unique(src_nodes)
        for node in unique_src:
            mask = (src_nodes == node)
            attention_weights[mask] = F.softmax(r_ij[mask].clone(), dim=0)

        # 加权聚合邻居特征
        weighted_h_j = attention_weights * h_j[dst_nodes]

        # 这里我们更新目标节点的特征（即高管/经理）
        updated_h_j = torch.zeros_like(h_j)
        #updated_h_j.index_add_(0, dst_nodes, weighted_h_j)
        updated_h_j = updated_h_j.index_add(0, dst_nodes, weighted_h_j)

        # 非线性激活
        updated_h_j = self.delta(updated_h_j)

        return updated_h_j

'''
class HeteGNN(MessagePassing):
    def __init__(self, input_dim, output_dim, rel_num, negative_slope=0.2):
        super(HeteGNN, self).__init__(aggr='add')  # Simplified initialization
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rel_num = rel_num
        self.negative_slope = negative_slope
        self.proj_x = nn.Linear(input_dim, output_dim, bias=False)
        # Linear projections for company and person embeddings
        self.proj_com = nn.Linear(input_dim, output_dim, bias=False)
        self.proj_per = nn.Linear(input_dim, output_dim, bias=False)
        self.proj_com2 = nn.Linear(input_dim, output_dim, bias=False)
        self.proj_per2 = nn.Linear(input_dim, output_dim, bias=False)
        #self.get_heter_rel = GetHeterRel(input_dim, output_dim, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),    ##目标是获取超图的隐藏关系
        #                                     residual=True)

        self.bn = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        # 定义 rel_wi，用于不同边类型的线性变换
        self.rel_wi = nn.ModuleList([nn.Linear(output_dim*2, output_dim) for _ in range(4)])

    def forward(self, x, heter_adj, matrix, rela_list_bi=None):
        num_firms = x.shape[0]
        # 创建 heter_adj 的深拷贝，防止原始对象被修改
        heter_adj_copy = copy.deepcopy(heter_adj)

        # get_heter_rel = self.get_heter_rel(x)
        if rela_list_bi:
            heter_adj_copy.append(torch.tensor(rela_list_bi, device=x.device))  # 将 get_heter_rel 也放到同一设备上
        # heter_adj.append(get_heter_rel)
        # Project the input embeddings
        e_f_matrix = matrix[0].T
        e_f_matrix = e_f_matrix.float()  # 将 e_f_matrix 转换为浮点数类型
        e_f = (e_f_matrix @ x)  # inter是高管-公司矩阵[667,163]，company_emb为[163,F]，目的是表示出高管的初始特征
        company_emb = self.proj_com(x)
        exe_emb = self.proj_per(e_f)
        emb = torch.cat((company_emb, exe_emb), dim=0)
        emb = self.bn(emb)  # torch.Size([830, 8])
        e_f_list = []
        for i in range(len(heter_adj_copy[:2])):  # 总共有3种异质边，高管-经理，高管-公司，公司与公司的隐藏关系
            heter_adj_copy[i] = torch.stack([heter_adj_copy[i][:, 0], heter_adj_copy[i][:, 1]], dim=0)
            rs = F.leaky_relu(
                self.propagate(heter_adj_copy[i], x=emb, edge_weight=torch.ones(len(heter_adj_copy[i][0]), 1),
                               edge_type=i),
                self.negative_slope)
            e_f_list.append(rs[:num_firms,])


        s_f_matrix = matrix[1]
        s_f_matrix = s_f_matrix.float()  # 将 e_f_matrix 转换为浮点数类型
        s_f = (s_f_matrix @ x)  # inter是高管-公司矩阵[667,163]，company_emb为[163,F]，目的是表示出高管的初始特征
        company_emb = self.proj_com2(x)
        sha_emb = self.proj_per2(s_f)
        emb2 = torch.cat((company_emb, sha_emb), dim=0)
        emb2 = self.bn2(emb2)  # torch.Size([830, 8])
        # Process edges based on their types
        s_f_list = []

        heter_adj_copy[2] = torch.stack([heter_adj_copy[2][:, 0], heter_adj_copy[2][:, 1]], dim=0)
        rs2 = F.leaky_relu(
            self.propagate(heter_adj_copy[2], x=emb2, edge_weight=torch.ones(len(heter_adj_copy[2][0]), 1),
                            edge_type=2),
            self.negative_slope)
        s_f_list.append(rs2[:num_firms,])

        if rela_list_bi:
            hidden_list = []
            x_proj = self.proj_x(x)
            heter_adj_copy[3] = torch.stack([heter_adj_copy[3][:, 0], heter_adj_copy[3][:, 1]], dim=0)
            rs3 = F.leaky_relu(
                self.propagate(heter_adj_copy[3], x=x_proj, edge_weight=torch.ones(len(heter_adj_copy[3][0]), 1),
                            edge_type=3),
                self.negative_slope)
            hidden_list.append(rs3)
            total_list = e_f_list+s_f_list+hidden_list
        else:
            total_list = e_f_list + s_f_list
        #return total_list, emb
        return e_f_list,emb

    def message(self, edge_index, x_i, x_j, edge_weight, edge_type):
        if torch.sum(edge_weight) != edge_index.shape[1]:
            x_j = self.rel_wi[edge_type](x_j)
            edge_weight = softmax(edge_weight, edge_index[1])
            rs = x_j * edge_weight
        else:
            node_f = torch.cat((x_i, x_j), 1)  # Concatenate node features
            temp = self.rel_wi[edge_type](node_f).to(x_i.device)  # Transform features
            alpha = softmax(temp, edge_index[1])  # Compute attention weights
            rs = x_j * alpha
        return rs

    def update(self, inputs):
        # Update node embeddings
        return super().update(inputs)
'''
#这个代码注释的原因是我发现heter_adj每一次都会发生修改，导致进入这个函数的原始heter_adj每次都不一样，上面那个代码就是修改好的利用深度复制，并且均在cuda里
'''   
class HeteGNN(MessagePassing):
    def __init__(self, input_dim, output_dim, rel_num, negative_slope=0.2):
        super(HeteGNN, self).__init__(aggr='add')  # Simplified initialization
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rel_num = rel_num
        self.negative_slope = negative_slope

        # Linear projections for company and person embeddings
        self.proj_com = nn.Linear(input_dim, output_dim, bias=False)
        self.proj_per = nn.Linear(input_dim, output_dim, bias=False)
        #self.get_heter_rel = GetHeterRel(input_dim, output_dim, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),    ##目标是获取超图的隐藏关系
        #                                     residual=True)

        self.bn = nn.BatchNorm1d(output_dim)
        # 定义 rel_wi，用于不同边类型的线性变换
        self.rel_wi = nn.ModuleList([nn.Linear(output_dim*2, output_dim) for _ in range(3)])

    def forward(self, x, heter_adj,e_f_matrix,rela_list_bi):
        #get_heter_rel = self.get_heter_rel(x)
        if rela_list_bi:
            heter_adj.append(torch.tensor(rela_list_bi, device=x.device))  # 将 get_heter_rel 也放到同一设备上
        #heter_adj.append(get_heter_rel)
        # Project the input embeddings
        e_f_matrix = e_f_matrix.T
        e_f_matrix = e_f_matrix.float()  # 将 e_f_matrix 转换为浮点数类型
        person = (e_f_matrix @ x) #inter是高管-公司矩阵[667,163]，company_emb为[163,F]，目的是表示出高管的初始特征
        company_emb = self.proj_com(x)
        person_emb = self.proj_per(person)
        emb = torch.cat((company_emb, person_emb), dim=0)
        emb = self.bn(emb) #torch.Size([830, 8])

        # Process edges based on their types
        rs_list = []

        for i in range(len(heter_adj)):  # 总共有3种异质边，高管-经理，高管-公司，公司与公司的隐藏关系
            #print(len(heter_adj))
            heter_adj[i] = torch.stack([heter_adj[i][:, 0], heter_adj[i][:, 1]], dim=0)
            rs = F.leaky_relu(self.propagate(heter_adj[i], x=emb, edge_weight=torch.ones(len(heter_adj[i][0]), 1), edge_type=i),
                                  self.negative_slope)
            rs_list.append(rs)
        if rela_list_bi:
            heter_adj.pop(2)
        return rs_list ,emb

    def message(self, edge_index, x_i, x_j, edge_weight, edge_type):
        if torch.sum(edge_weight) != edge_index.shape[1]:
            x_j = self.rel_wi[edge_type](x_j)
            edge_weight = softmax(edge_weight, edge_index[1])
            rs = x_j * edge_weight
        else:
            node_f = torch.cat((x_i, x_j), 1)  # Concatenate node features
            temp = self.rel_wi[edge_type](node_f).to(x_i.device)  # Transform features
            alpha = softmax(temp, edge_index[1])  # Compute attention weights
            rs = x_j * alpha
        return rs

    def update(self, inputs):
        # Update node embeddings
        return super().update(inputs)

'''


import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import NMF
def check_nan(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")

class GetHyperRel(nn.Module):   # 目标是获取超图的隐藏关系
    def __init__(self,
                 in_channel,
                 out_sz,
                 n_components=30,
                 threshold=0.7,
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=None,
                 residual=False):
        super(GetHyperRel, self).__init__()
        self.in_channel = in_channel
        self.out_sz = out_sz
        self.n_components = n_components
        self.threshold = threshold
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.linear = nn.Linear(n_components, 8)

    def forward(self, x):
        # 转置并添加批次维度
        seq = x.transpose(0, 1).unsqueeze(0)  # 转换为 [1, 特征数, 节点数]

        # 卷积操作
        seq_fts = self.conv1(seq)  # [1, out_sz, 节点数]

        f_1 = self.conv2_1(seq_fts)  # [1, 1, 节点数]
        f_2 = self.conv2_2(seq_fts)  # [1, 1, 节点数]

        # 计算 logits
        logits = f_1 + torch.transpose(f_2, 2, 1)  # [1, 节点数, 节点数]
        # 找到 logits 的最小值和最大值
        # min_val = torch.min(logits)
        # max_val = torch.max(logits)
        #
        # # 进行最小-最大归一化，将 logits 归一化到 [0, 1]
        # logits = (logits - min_val) / (max_val - min_val)
        # 使用 ReLU 处理负值
        logits = self.relu(logits)


        # Convert logits to numpy for NMF
        logits_np = logits.squeeze().detach().cpu().numpy()  # 移除批次维度并转换为 numpy 数组
        #logits_np = np.clip(logits_np,1e-4,None)
        if np.isnan(logits_np).any():
            print("Found NaN in logits_np")
        # Apply NMF
        model = NMF(n_components=self.n_components,init='random')
        W = model.fit_transform(logits_np)

        # Normalize W matrix to [0, 1] range
        W_min = W.min(axis=0)
        W_max = W.max(axis=0)
        W = (W - W_min) / (W_max - W_min + 1e-6)
        check_nan(torch.tensor(W), "W matrix after normalization")

        # Generate groups from W matrix
        rela_list = []
        for i in range(W.shape[1]):
            group = np.where(W[:, i] > self.threshold)[0].tolist()
            rela_list.append(group)

        # 分类 rela_list
        rela_list_bi = [group for group in rela_list if len(group) == 2]  # 只包含长度为2的关系
        rela_list_mul = [group for group in rela_list if len(group) > 2]  # 包含长度大于2的关系

        # 使用 rela_list_mul 生成新矩阵 [节点数, len(rela_list_mul)]
        n_components_mul = len(rela_list_mul)
        batch_size = x.shape[0]
        group_matrix = np.zeros((batch_size, n_components_mul), dtype=int)

        # 填充矩阵，group_matrix[group, j] = 1
        for j, group in enumerate(rela_list_mul):
            if len(group) > 0:  # 确保 group 不为空
                group_matrix[group, j] = 1

        # 返回 rela_list_bi, rela_list_mul, 和转换后的矩阵
        return rela_list_bi, rela_list_mul, torch.tensor(group_matrix, dtype=torch.float32)



import torch
import numpy as np
from sklearn.decomposition import NMF


import torch
import numpy as np
from sklearn.decomposition import NMF
import torch.nn as nn

import torch
import numpy as np
from sklearn.decomposition import NMF




class heter_edge_relation_att(nn.Module): #多头注意力机制
    def __init__(self, nfeat, nhid, dropout,use_hidden_rel):
        super(heter_edge_relation_att, self).__init__()
        self.use_hidden_rel = use_hidden_rel
        self.output_dim = nhid
        self.n_relations = 4 if use_hidden_rel == 1 else 3
        self.n_heads = 8  # 默认多头数量，可以根据需要调整
        self.x_pro = nn.Linear(nfeat, nhid)  # 最终特征的变换
        # 在 __init__ 中初始化 Q, K, V 的线性层
        self.cq_linears = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(self.n_relations)])  # Query layers
        self.ck_linears = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(self.n_relations)])  # Key layers
        self.cv_linears = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(self.n_relations)])  # Value layers

        self.fi_linear = nn.Linear(nhid, nhid)  # 最终特征的变换
        self.crelation_pri = nn.Parameter(torch.ones(self.n_relations))  # 每种关系类型的权重
        self.skip = nn.Parameter(torch.ones(1))  # Skip connection parameter
        self.bn = nn.BatchNorm1d(nhid)  # Batch normalization

    def forward(self, x, re_list, n_heads):
        com_att = []  # 用于存储注意力权重
        v_values = []  # 用于存储值

        # 确保输入 x 在同一个设备上
        device = x.device
        x_pro = self.x_pro(x)
        for i in range(len(re_list)):
            # 获取当前关系的嵌入并移动到相同设备
            rel_emb = re_list[i].to(device)

            # 将线性层参数和输入移动到同一设备
            self.cq_linears[i] = self.cq_linears[i].to(device)
            self.ck_linears[i] = self.ck_linears[i].to(device)
            self.cv_linears[i] = self.cv_linears[i].to(device)

            q_mat = self.cq_linears[i](x_pro)  # 从输入特征生成 Query
            k_mat = self.ck_linears[i](rel_emb)  # 从关系嵌入生成 Key
            v_mat = self.cv_linears[i](rel_emb)  # 从关系嵌入生成 Value

            # 将 Q, K, V 分成多头
            q_mat = q_mat.view(q_mat.size(0), self.n_heads, -1)  # [758, n_heads, nhid/n_heads]
            k_mat = k_mat.view(k_mat.size(0), self.n_heads, -1)
            v_mat = v_mat.view(v_mat.size(0), self.n_heads, -1)

            # QK 点积并进行缩放
            att_score = torch.matmul(q_mat, k_mat.transpose(-1, -2)) * self.crelation_pri[i] / math.sqrt(
                self.output_dim)
            att_weights = F.softmax(att_score, dim=-1)  # softmax 计算注意力权重

            # 收集注意力权重和 Value
            com_att.append(att_weights)
            v_values.append(v_mat)

        res = 0
        for i in range(len(re_list)):
            # 根据注意力权重对 Value 加权求和
            res += torch.matmul(com_att[i], v_values[i])

        # 将 res 重新调整形状，以便于后续处理
        res = res.view(res.size(0), -1)  # [758, nhid]，视具体情况调整
        # Skip connection 和 GELU 激活函数
        alpha = torch.sigmoid(self.skip.to(device))  # 确保 skip 参数在同一设备上
        res = (self.fi_linear(res.to(device)) + alpha * F.gelu(x_pro))

        return res


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class hyper_edge_relation_att(nn.Module):  # Multi-head attention mechanism
    def __init__(self, nfeat, nhid, dropout, use_hidden_rel):
        super(hyper_edge_relation_att, self).__init__()

        self.output_dim = nhid
        self.n_heads = 8  # Default number of heads, can be adjusted
        self.use_hidden_rel = use_hidden_rel

        # Dynamically set the number of relations based on use_hidden_rel
        self.n_relations = 3 if use_hidden_rel == 1 else 2

        # Initialize Q, K, V linear layers based on n_relations
        self.cq_linears = nn.ModuleList([nn.Linear(nfeat, nhid) for _ in range(self.n_relations)])  # Query layers
        self.ck_linears = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(self.n_relations)])  # Key layers
        self.cv_linears = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(self.n_relations)])  # Value layers

        self.fi_linear = nn.Linear(nfeat, nhid)  # Final feature transformation
        self.crelation_pri = nn.Parameter(torch.ones(self.n_relations))  # Weight for each relation type
        self.skip = nn.Parameter(torch.ones(1))  # Skip connection parameter
        self.bn = nn.BatchNorm1d(nhid)  # Batch normalization

    def forward(self, x, re_list, n_heads):
        com_att = []  # Store attention weights
        v_values = []  # Store values

        # Ensure input x is on the same device
        device = x.device
        n_relations = self.n_relations  # Number of relations based on use_hidden_rel
        re_list = list(re_list)
        for i in range(n_relations):
            # Get the embedding for the current relation and move it to the same device
            rel_emb = re_list[i].to(device)

            # Move the linear layers to the same device
            self.cq_linears[i] = self.cq_linears[i].to(device)
            self.ck_linears[i] = self.ck_linears[i].to(device)
            self.cv_linears[i] = self.cv_linears[i].to(device)

            # Generate Q, K, V matrices
            q_mat = self.cq_linears[i](x)  # Query from input features
            k_mat = self.ck_linears[i](rel_emb)  # Key from relation embedding
            v_mat = self.cv_linears[i](rel_emb)  # Value from relation embedding

            # Split Q, K, V into multiple heads
            q_mat = q_mat.view(q_mat.size(0), self.n_heads, -1)  # [batch_size, n_heads, nhid/n_heads]
            k_mat = k_mat.view(k_mat.size(0), self.n_heads, -1)
            v_mat = v_mat.view(v_mat.size(0), self.n_heads, -1)

            # Compute scaled dot-product attention
            att_score = torch.matmul(q_mat, k_mat.transpose(-1, -2)) * self.crelation_pri[i] / math.sqrt(
                self.output_dim)
            att_weights = F.softmax(att_score, dim=-1)  # Softmax to compute attention weights

            # Store attention weights and values
            com_att.append(att_weights)
            v_values.append(v_mat)

        # Compute the weighted sum of values using attention weights
        res = 0
        for i in range(n_relations):
            res += torch.matmul(com_att[i], v_values[i])

        # Reshape res for subsequent processing
        res = res.view(res.size(0), -1)  # [batch_size, nhid]

        # Apply skip connection and GELU activation
        alpha = torch.sigmoid(self.skip.to(device))  # Ensure skip parameter is on the same device
        res = (res + alpha * F.gelu(self.fi_linear(x)))  # Skip connection with activation

        # Optionally apply batch normalization
        res = self.bn(res)

        return res  # Return the processed result


import torch
import torch.nn as nn
import torch.nn.functional as F

#假如现在超图的列表中有两个元素【A,B】，维度分别为N*F，异质图列表有3个元素[C,D,E]，维度分别为N*F，现在首先做注意力机制，对于超图而言的注意力权重，就用拼接AC，AD，AE以及BC，BD，BE，那么超图A元素就可以重新加权表示了（利用AC，AD，AE）；
# 超图B元素也可以重新加权表示了，懂了吗？然后，对于异质图，本质上对于C的加权就使用CA和CB，
# 对于D，E也是类似的，这就是我用的注意力机制，值得注意的是，新的ABCDE这5个元素都是重新加权（运用注意力机制）了的，然后维度还是变成N*F，最后一步的双向加权是指总的特征表示为（A+B）*len（超图）+（C+D+E）*len（异质图），再将这个式子整的除以二
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModalAttention, self).__init__()
        # Linear layer for attention score
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, hetero_list, hyper_list):
        """
        Forward pass to calculate attention scores and perform weighted sum of hypergraph features.
        :param hetero_list: List of [N, F] tensors from the heterogeneous graph
        :param hyper_list: List of [N, F] tensors from the hypergraph
        :return: Final interaction features of shape [N, F]
        """
        # Initialize new feature lists
        new_hyper_list = []
        new_hetero_list = []

        # Compute new hypergraph features
        for h_hyper in hyper_list:
            weighted_sum = torch.zeros_like(h_hyper)
            for h_hetero in hetero_list:
                concat_features = torch.cat([h_hyper, h_hetero], dim=-1)  # [N, 2*F]
                attention_score = self.leakyrelu(self.attention_fc(concat_features))
                attention_weight = F.softmax(attention_score, dim=0)  # Softmax normalization

                weighted_sum += attention_weight * h_hetero  # Re-weight the hetero features
            new_hyper_list.append(weighted_sum)  # New representation for each hypergraph feature

        # Compute new heterogeneous features
        for h_hetero in hetero_list:
            weighted_sum = torch.zeros_like(h_hetero)
            for h_hyper in hyper_list:
                concat_features = torch.cat([h_hetero, h_hyper], dim=-1)  # [N, 2*F]
                attention_score = self.leakyrelu(self.attention_fc(concat_features))
                attention_weight = F.softmax(attention_score, dim=0)  # Softmax normalization

                weighted_sum += attention_weight * h_hyper  # Re-weight the hyper features
            new_hetero_list.append(weighted_sum)  # New representation for each heterogeneous feature

        # Aggregate final representation
        h_final = (sum(new_hyper_list) / len(hyper_list) + sum(new_hetero_list) / len(
            hetero_list)) / 2  # Final representation

        return h_final


class GraphInteractionModel(nn.Module):
    def __init__(self, hidden_dim):
        super(GraphInteractionModel, self).__init__()
        # Initialize the cross-modal attention mechanism
        self.attention_mechanism = CrossModalAttention(hidden_dim)

    def forward(self, hetero_list, hyper_list):
        """
        Forward pass for the bi-directional interaction model.
        :param hetero_list: List of [N, F] tensors from the heterogeneous graph
        :param hyper_list: List of [N, F] tensors from the hypergraph
        :return: Final bi-directional interaction features of shape [N, F]
        """
        # Perform cross-modal attention and compute interaction features
        h_final = self.attention_mechanism(hetero_list, hyper_list)

        return h_final

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModalAttention, self).__init__()
        # Linear layer for attention score
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, hetero_list, hyper_list):
        """
        Forward pass to calculate attention scores and perform weighted sum of hypergraph features.
        :param hetero_list: List of [N, F] tensors from the heterogeneous graph
        :param hyper_list: List of [N, F] tensors from the hypergraph
        :return: Final interaction features of shape [N, F]
        """
        # Initialize attention weights lists
        attention_weights_he = []
        attention_weights_hy = []

        # Compute attention weights between each pair of heterogeneous and hypergraph features
        for h_hetero in hetero_list:
            temp_attention = []
            for h_hyper in hyper_list:
                # Concatenate heterogeneous and hypergraph features [N, 2*F]
                concat_features = torch.cat([h_hetero, h_hyper], dim=-1)
                # Calculate attention scores [N, 1]
                attention_score = self.leakyrelu(self.attention_fc(concat_features))
                temp_attention.append(attention_score)

            # Stack the attention scores for current heterogeneous feature [N, L]
            temp_attention = torch.cat(temp_attention, dim=1)
            temp_attention = temp_attention.float()
            attention_weights_he.append(temp_attention)

        # Convert attention weights list into a tensor [K, N, L]
        attention_weights_he = torch.stack(attention_weights_he, dim=0).float()
        # Apply softmax along the hypergraph dimension (L dimension)
        attention_weights_he = F.softmax(attention_weights_he, dim=2)

        # Perform the attention-weighted sum of hypergraph features
        h_interaction = []
        hyper_tensor = torch.stack(hyper_list, dim=0)  # [L, N, F]

        for i, h_hetero in enumerate(hetero_list):
            # Weighted sum across the hypergraph for each heterogeneous feature
            weighted_hyper = torch.einsum('nl,nlf->nf', attention_weights_he[i], hyper_tensor.permute(1, 0, 2))
            # Normalize by number of hypergraph features
            weighted_hyper = weighted_hyper / len(hyper_list)
            h_interaction.append(weighted_hyper)

        # Compute attention weights for hypergraph features
        for h_hyper in hyper_list:
            temp_attention = []
            for h_hetero in hetero_list:
                # Concatenate hypergraph and heterogeneous features [N, 2*F]
                concat_features = torch.cat([h_hyper, h_hetero], dim=-1)
                # Calculate attention scores [N, 1]
                attention_score = self.leakyrelu(self.attention_fc(concat_features))
                temp_attention.append(attention_score)

            # Stack the attention scores for current hypergraph feature [N, L]
            temp_attention = torch.cat(temp_attention, dim=1)
            temp_attention = temp_attention.float()
            attention_weights_hy.append(temp_attention)

        # Convert attention weights list into a tensor [L, N, K]
        attention_weights_hy = torch.stack(attention_weights_hy, dim=0).float()
        # Apply softmax along the heterogeneous graph dimension (K dimension)
        attention_weights_hy = F.softmax(attention_weights_hy, dim=2)

        # Perform the attention-weighted sum of heterogeneous features
        for j, h_hyper in enumerate(hyper_list):
            # Weighted sum across the heterogeneous graph for each hypergraph feature
            weighted_hetero = torch.einsum('nl,nlf->nf', attention_weights_hy[j],
                                           torch.stack(hetero_list, dim=0).permute(1, 0, 2))
            # Normalize by number of heterogeneous features
            weighted_hetero = weighted_hetero / len(hetero_list)
            h_interaction.append(weighted_hetero)

        # Sum the interaction terms from all features [N, F]
        h_final = sum(h_interaction) / len(h_interaction)

        return h_final


class GraphInteractionModel(nn.Module):
    def __init__(self, hidden_dim):
        super(GraphInteractionModel, self).__init__()
        # Initialize the cross-modal attention mechanism
        self.attention_mechanism = CrossModalAttention(hidden_dim)

    def forward(self, hetero_list, hyper_list):
        """
        Forward pass for the bi-directional interaction model.
        :param hetero_list: List of [N, F] tensors from the heterogeneous graph
        :param hyper_list: List of [N, F] tensors from the hypergraph
        :return: Final bi-directional interaction features of shape [N, F]
        """
        # Perform cross-modal attention and compute interaction features
        h_final = self.attention_mechanism(hetero_list, hyper_list)

        return h_final
'''

'''
# Cross-modal attention mechanism for bi-directional information interaction
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)  # Linear layer for attention score
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, hetero_list, hyper_list):
        """
        Forward pass to calculate attention scores and perform weighted sum of hypergraph features.
        :param hetero_list: List of [N, F] tensors from the heterogeneous graph
        :param hyper_list: List of [N, F] tensors from the hypergraph
        :return: Final interaction features of shape [N, F]
        """
        #N, F = hetero_list[0].size(0), hetero_list[0].size(1)  # Get node count and feature dimension

        # Compute attention weights between each pair of heterogeneous and hypergraph features
        attention_weights = []
        for h_hetero in hetero_list:
            temp_attention = []
            for h_hyper in hyper_list:
                # Concatenate heterogeneous and hypergraph features [N, 2*F]
                concat_features = torch.cat([h_hetero, h_hyper], dim=-1)
                # Calculate attention scores [N, 1]
                attention_score = self.leakyrelu(self.attention_fc(concat_features))
                temp_attention.append(attention_score)

            # Stack the attention scores for current heterogeneous feature [N, L]
            temp_attention = torch.cat(temp_attention, dim=1)  # Change from stack to cat
            # Ensure temp_attention is float tensor
            temp_attention = temp_attention.float()
            # Store attention weights for each heterogeneous feature
            attention_weights.append(temp_attention)

        # Convert attention weights list into a tensor [K, N, L] where K is the number of hetero features
        attention_weights = torch.stack(attention_weights, dim=0).float()  # Ensure float type
        # Apply softmax along the hypergraph dimension (L dimension)
        attention_weights = F.softmax(attention_weights, dim=2)

        # Perform the attention-weighted sum of hypergraph features
        h_interaction = []
        hyper_tensor = torch.stack(hyper_list, dim=0)  # [L, N, F]
        for i, h_hetero in enumerate(hetero_list):
            # Weighted sum across the hypergraph for each heterogeneous feature
            weighted_hyper = torch.einsum('nl,nlf->nf', attention_weights[i], hyper_tensor.permute(1, 0, 2))

            h_interaction.append(weighted_hyper)

        # Sum the interaction terms from all heterogeneous features [N, F]
        h_final = sum(h_interaction) / len(h_interaction)

        return h_final


# Bi-directional interaction model
class GraphInteractionModel(nn.Module):
    def __init__(self, hidden_dim):
        super(GraphInteractionModel, self).__init__()
        # Initialize the cross-modal attention mechanism
        self.attention_mechanism = CrossModalAttention(hidden_dim)

    def forward(self, hetero_list, hyper_list):
        """
        Forward pass for the bi-directional interaction model.
        :param hetero_list: List of [N, F] tensors from the heterogeneous graph
        :param hyper_list: List of [N, F] tensors from the hypergraph
        :return: Final bi-directional interaction features of shape [N, F]
        """
        # Perform cross-modal attention and compute interaction features
        h_final = self.attention_mechanism(hetero_list, hyper_list)

        return h_final '''
