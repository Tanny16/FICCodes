''' Define the HGTAN model '''
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
#from HGTAN.tri_attn import HGAT
#from HGTAN.temp_layers import TemporalAttention
import torch
import os
import torch.nn as nn
from training.mytools import *
import torch.nn.functional as F
from HGTAN.mylayers import HeteGNN,GetHyperRel,hyper_edge_relation_att,heter_edge_relation_att,GraphInteractionModel





class myHGTAN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            num_stock, rnn_unit, n_hid, n_class,
            feature,
            d_word_vec, d_model,
            n_head, d_k, d_v, dropout,
            tgt_emb_prj_weight_sharing,use_hidden_rel,use_interaction):

        super().__init__()
        self.dropout = dropout
        self.linear = nn.Linear(feature, d_word_vec)
        self.use_hidden_rel = use_hidden_rel
        self.use_interaction = use_interaction
        self.n_hid=n_hid
        # torch.nn.init.xavier_uniform_(self.linear.weight)
        # if self.linear.bias is not None:
        #     torch.nn.init.zeros_(self.linear.bias)
        '''
        self.rnn2 = nn.GRU(feature,
                          rnn_unit,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)'''
        self.rnn = Graph_GRUModel(num_stock, feature, rnn_unit)
        #self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)

        self.HGAT = HGAT(nfeat=rnn_unit,
                         nhid=n_hid,
                         dropout=dropout,
                         use_hidden_rel=use_hidden_rel,
                         use_interaction = use_interaction

                         )

        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        #self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        #nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1

        self.bn = nn.BatchNorm1d(n_hid)  # Batch normalization
        self.bn_in = nn.BatchNorm1d(feature)

    def forward(self,src_seq1, hyper_adj,heter_adj,matrix):
        stock_num = src_seq1.size(1)
        seq_len = src_seq1.size(0)
        dim = src_seq1.size(2)
        #src_seq = apply_batch_norm(src_seq1)
        src_seq1_flat = src_seq1.view(-1, dim)
        src_seq = self.bn_in(src_seq1_flat)
        src_seq = src_seq.view(seq_len, stock_num, -1)
        if torch.isnan(src_seq).any():
            print("src_seq1 中存在 NaN 值！")
        #src_seq = self.linear(src_seq_x) #src_seq [10, 758, 10] 变成   [10, 758, 16]
        #src_seq=src_seq1

        #src_seq = torch.reshape(src_seq, (stock, seq_len, dim))  #变成[488, 10, 16]

        rnn_output = self.rnn(src_seq)
        rnn_output = F.dropout(rnn_output, self.dropout, training=self.training)
        '''
        rnn_output, *_ = self.rnn2(src_seq)  #torch.Size([758, 10, 32])
        rnn_output=rnn_output[:,-1,:]  #torch.Size([488, 32])
        rnn_output = F.dropout(rnn_output, self.dropout ,training=self.training)'''

        if torch.isnan(rnn_output).any():
            print("rnn_output 中存在 NaN 值！")
        #enc_output = torch.reshape(rnn_output, (seq_len, stock)) # [758, 32]

        HGAT_output = self.HGAT(rnn_output, hyper_adj,heter_adj,matrix)   #输入gru编码后的enc_output [32, 758, 32]   输出[32, 758, 8]
        HGAT_output = self.bn(HGAT_output)
        # 拼接 HGAT_output 和 rnn_output，沿着最后一个维度（维度 1）拼接
        #combined_output = torch.cat((rnn_output, HGAT_output), dim=1) #HGAT_output.shape torch.Size([163, 8]) rnn_output.shape torch.Size([163, 32])

        #seq_logit = self.tgt_word_prj(combined_output)
        #seq_logit = self.tgt_word_prj(HGAT_output)
        seq_logit = F.elu(self.tgt_word_prj(HGAT_output ))
        output = F.log_softmax(seq_logit, dim=1)

        return output




class Graph_GRUModel(nn.Module):#Graph_GRUModel 类继承自 nn.Module，表示一个图卷积循环神经网络（Graph GRU Model）。该模型用于处理图数据，其中包含 num_nodes 个节点，每个节点的输入维度为 input_dim，并且具有 hidden_dim 个隐藏单元。
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(Graph_GRUModel, self).__init__()  #在初始化方法中，首先调用了父类 nn.Module 的初始化方法，确保正确初始化模型。
        self.hidden_dim = hidden_dim  #将传入的 hidden_dim 参数保存为模型的属性，表示模型的隐藏单元数量。
        self.gru_cell = Graph_GRUCell(num_nodes, input_dim, hidden_dim) #调用grucell方法
        self.reset_parameters() #重置参数

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None): #定义了前向传播方法 forward，它接受输入 x 和可选的隐藏状态 hidden，并返回模型的输出。
        if hidden is None: #检查是否提供了隐藏状态 hidden。如果没有提供，就执行以下操作：
            #创建一个全零的隐藏状态张量 hidden，其形状为 (x.size()[1], self.hidden_dim)，数据类型和计算设备与输入 x 保持一致。
            hidden = torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype)
        #循环遍历输入数据 x 的每个时间步 seq，并在每个时间步上调用 self.gru_cell 来更新隐藏状态 hidden。
        for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)
            #最终返回最后一个时间步的隐藏状态 hidden 作为模型的输出。
        return hidden
        #总的来说，这个模型是一个循环神经网络（GRU），接受输入数据 x，并可选地接受初始隐藏状态 hidden。它通过循环处理输入数据的每个时间步，更新隐藏状态，并返回最终的隐藏状态作为模型的输出。模型内部使用了一个 Graph_GRUCell 子模型来执行单个时间步的操作。

class Graph_GRUCell(nn.Module): #定义了一个名为 Graph_GRUCell 的自定义 PyTorch 模块，用于实现一个特殊的循环神经网络（GRU）单元。这个模块的主要功能是实现 GRU 单元的前向传播操作，包括门控信息的计算和新的隐藏状态的更新。
    def __init__(self, num_nodes, input_size, hidden_size, bias=True): #构造函数，初始化模块的属性。
        super(Graph_GRUCell, self).__init__()
        # num_nodes：节点数量。
        # input_size：输入数据的特征维度。
        # hidden_size：隐藏状态的特征维度。
        # bias：一个布尔值，表示是否使用偏差项（默认为True）。
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        #这是两个 Graph_Linear 模块，分别用于处理输入 x 和隐藏状态 hidden。
        self.x2h = Graph_Linear(num_nodes, input_size, 3 * hidden_size, bias=bias)
        self.h2h = Graph_Linear(num_nodes, hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()#一个自定义方法，用于重新初始化模块的参数。
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x, hidden): #前向传播方法，接受输入 x 和上一个时刻的隐藏状态 hidden。
        #分别通过 self.x2h 和 self.h2h 处理输入 x 和隐藏状态 hidden，生成门控信息。
        gate_x = self.x2h(x)  #gate_x 和 gate_h 的形状都是 (num_nodes, 3 * hidden_size)。然后，将它们展平为 (num_nodes, 3, hidden_size) 的形状。
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1) #将门控信息拆分成三个部分，即重置门、输入门和新门。
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r) #分别通过 torch.sigmoid 函数计算重置门和输入门。
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n)) #通过 torch.tanh 函数计算新门的候选值。
        hy = newgate + inputgate * (hidden - newgate) #通过组合新门和输入门以及上一个隐藏状态 hidden 的差异来计算新的隐藏状态
        return hy

class Graph_Linear(nn.Module): #这是一个自定义 PyTorch 模块的定义，该模块继承自 nn.Module 类，表示它是一个 PyTorch 模型。
    def __init__(self,num_nodes, input_size, hidden_size, bias=True):#这是模块的构造函数，用于初始化模块的各个属性。
        super(Graph_Linear, self).__init__()
        self.bias = bias  #一个布尔值，表示是否使用偏差项（默认为 True）。将构造函数中传入的 bias 参数存储为模块的属性，以便后续使用。
        self.W = nn.Parameter(torch.zeros(num_nodes,input_size,hidden_size))#创建一个可训练的参数张量 self.W，表示权重矩阵，其维度为 (num_nodes, input_size, hidden_size)，并将其初始化为零。
        self.b = nn.Parameter(torch.zeros(num_nodes,hidden_size))#创建一个可训练的参数张量 self.b，表示偏差项，其维度为 (num_nodes, hidden_size)，并将其初始化为零。
        self.reset_parameters()#调用 reset_parameters 方法来重新初始化模块的参数。这是一个自定义的方法，通常用于在模块创建时初始化参数。
    def reset_parameters(self):#这是一个自定义方法，用于重新初始化模块的参数。
        reset_parameters(self.named_parameters)
    def forward(self, x):#这是模块的前向传播方法。它接受输入张量 x，并根据模型的权重矩阵 self.W 和偏差项 self.b 来计算输出。
        output = torch.bmm(x.unsqueeze(1), self.W)#首先，将输入 x 进行一些维度操作，然后使用 torch.bmm 函数进行矩阵相乘，得到输出 output。
        output = output.squeeze(1)#对输出 output 进行维度压缩，以去除不必要的维度。
        if self.bias:#如果模块中设置了使用偏差项，将偏差项 self.b 添加到输出中。
            output = output + self.b
        return output





import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from HGTAN.mylayers import edge_attn
import math
import time
class HGAT(nn.Module):
    """Tri-attention Modules."""

    def __init__(self, nfeat, nhid, dropout ,use_hidden_rel,use_interaction):

        super(HGAT, self).__init__()
        self.use_hidden_rel = use_hidden_rel
        self.use_interaction = use_interaction
        self.hyperedge = hyperedge_attn(nfeat, nhid, dropout,use_hidden_rel)
        self.attn = edge_attn(nhid, nhid*2)
        self.dropout = dropout
        self.output_dim = nhid
        self.ck_linears = nn.ModuleList()
        self.cq_linears = nn.ModuleList()
        self.hyper_relation_att = hyper_edge_relation_att(nfeat, nhid, dropout,use_hidden_rel)
        self.hete_relation_att = heter_edge_relation_att(nfeat, nhid, dropout,use_hidden_rel)
        self.hete_emb = HeteGNN(nfeat, nhid, dropout)
        for t in range(2): #注意这个2代表有两种超图关系
            self.ck_linears.append(nn.Linear(nhid, nhid))
            self.cq_linears.append(nn.Linear(nfeat, nhid))

        self.cv_linear = nn.Linear(nhid, nhid)
        self.crelation_pri = nn.Parameter(torch.ones(2)) #生成与关系种类数相同的1张量
        self.GraphInteractionModel = GraphInteractionModel(nhid)
        self.bn = nn.BatchNorm1d(nhid)


        self.skip = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(nhid)
        self.GraphLevelAttention = GraphLevelAttention(nhid, nhid, dropout ,use_hidden_rel,use_interaction)
        self.hiden_hyper_rel = GetHyperRel(nfeat, nhid,
                                           n_components=int(os.environ.get('COMNC_HIDDEN_RELATIONS', '40')),
                                           threshold=float(os.environ.get('COMNC_HIDDEN_THRESHOLD', '0.8')),
                                           in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),  ##目标是获取超图的隐藏关系
                                           residual=True)
        #self.reset_parameters()
    def forward(self, x, hyper_adj,heter_adj,matrix): #x torch.Size([758, 32])  H.shape [758, 62]  nhid=8
        device = x.device
        num_company = x.shape[0]
        #x = F.dropout(x, self.dropout, training=self.training)  #x torch.Size([ 758, 32]) self.training=True
        if self.use_hidden_rel:
            rela_list_bi, rela_list_mul , rela_list_mul_ma= self.hiden_hyper_rel(x)  # 超图隐式关系推断,主要是用到这个hiden_hyper_rel
            re_list = self.hyperedge(x, hyper_adj, rela_list_mul_ma)
            hete_emb, emb = self.hete_emb(x, heter_adj, matrix, rela_list_bi)
        else:

        #这一步分别处理所有超图（包括推断的关系和预定义的关系）的超图聚合，主要包括两个组件超图隐式关系推断器和节点间注意力
            #start_time = time.time()
            re_list = self.hyperedge(x, hyper_adj)  #all_hyperedge_fts[758, 8]代表62个基金关系的嵌入  industry_tensor是行业关系嵌入[758, 8]

            hete_emb ,emb= self.hete_emb(x, heter_adj, matrix)
            #epoch_time = time.time() - start_time
        if self.use_interaction:
            interaction = self.GraphInteractionModel(hete_emb,list(re_list))
            interaction = self.bn(interaction)
            res_total = self.GraphLevelAttention(hete_emb, list(re_list), interaction)
            res_total = self.bn(res_total)
        else:
            res_total = self.GraphLevelAttention(hete_emb, list(re_list))
            res_total = self.bn(res_total)


        #res = self.hyper_relation_att(x,re_list,8)   #这一步处理超图的多个关系聚合器，即边的注意力（第二层注意力）
        #res = F.dropout(res, self.dropout, training=self.training)

        #现在处理异质图，第一步节点注意力和边注意力
        #heter_adj

        #print('2',len(heter_adj))
        #res2 = self.hete_relation_att(x, hete_emb, 8)
        #res2 = F.dropout(res2, self.dropout, training=self.training)

        #res_total = nn.GELU(self.GraphLevelAttention(res,res2[:num_company]))
        #gelu = nn.GELU()  # 实例化 GELU
        #res_total = self.GraphLevelAttention(res, res2[:num_company])
        #res_total = F.dropout(res_total, self.dropout, training=self.training)



        return res_total  #[32, 758, 8]






import torch
import torch.nn as nn
import torch.nn.functional as F




from HGTAN.mylayers import Attn_head, Attn_head_adj
class hyperedge_attn(nn.Module):
    def __init__(self, nfeat, nhid, dropout, use_hidden_rel):
        """ Intra-hyperedge attention module with optional hidden relationship usage. """

        super(hyperedge_attn, self).__init__()
        self.nhid = nhid
        self.intra_hpyeredge = Attn_head(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                        residual=True)
        self.intra_hpyeredge2 = Attn_head(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                         residual=True)
        self.intra_hpyeredge3 = Attn_head(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                          residual=True)
        self.hiden_hyper_rel = GetHyperRel(nfeat, nhid,
                                           n_components=int(os.environ.get('COMNC_HIDDEN_RELATIONS', '40')),
                                           threshold=float(os.environ.get('COMNC_HIDDEN_THRESHOLD', '0.8')),
                                           in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                             residual=True)
        self.dropout = dropout
        self.use_hidden_rel = use_hidden_rel  # New parameter to control the use of hidden relationships

    def forward(self, x, hyper_adj, rela_list_mul=None):
        stock = x.size(0)
        industry_adj = hyper_adj[0]
        region_adj = hyper_adj[1]

        # Process industry_adj
        single_edge = torch.zeros(stock, self.nhid).cuda()  # torch.Size([758, 8])
        for i in range(industry_adj.shape[1]):
            intra_hyperedge_fts = torch.randn(0).cuda()
            node_set = torch.nonzero(industry_adj[:, i], as_tuple=False)

            for j in range(len(node_set)):
                node_index = node_set[j]
                intra_hyperedge_fts = torch.cat([intra_hyperedge_fts, x[node_index, :]], dim=0)

            after_intra = self.intra_hpyeredge(intra_hyperedge_fts).squeeze(0)
            node_index = node_set.squeeze(1)
            single_edge[node_index, :] += after_intra

        # Process region_adj
        single_edge2 = torch.zeros(stock, self.nhid).cuda()
        for i in range(region_adj.shape[1]):
            intra_hyperedge_fts2 = torch.randn(0).cuda()
            node_set2 = torch.nonzero(region_adj[:, i], as_tuple=False)

            for j in range(len(node_set2)):
                node_index2 = node_set2[j]
                intra_hyperedge_fts2 = torch.cat([intra_hyperedge_fts2, x[node_index2, :]], dim=0)

            after_intra2 = self.intra_hpyeredge2(intra_hyperedge_fts2).squeeze(0)
            node_index2 = node_set2.squeeze(1)
            single_edge2[node_index2, :] += after_intra2

        result_set = {single_edge, single_edge2}  # Initialize the result set with single_edge and single_edge2

        # Optionally process rela_list_mul if use_hidden_rel is set to 1
        if self.use_hidden_rel:
            single_edge3 = torch.zeros(stock, self.nhid).cuda()
            for i in range(rela_list_mul.shape[1]):
                intra_hyperedge_fts3 = torch.randn(0).cuda()
                node_set3 = torch.nonzero(rela_list_mul[:, i], as_tuple=False)

                for j in range(len(node_set3)):
                    node_index3 = node_set3[j]
                    intra_hyperedge_fts3 = torch.cat([intra_hyperedge_fts3, x[node_index3, :]], dim=0)

                if len(intra_hyperedge_fts3.shape) >= 2:
                    after_intra3 = self.intra_hpyeredge3(intra_hyperedge_fts3).squeeze(0)
                    node_index3 = node_set3.squeeze(1)
                    single_edge3[node_index3, :] += after_intra3

            result_set.add(single_edge3)  # Add single_edge3 to the result set if use_hidden_rel is True

        return result_set



class GraphLevelAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout_rate=0.5, use_hidden_rel=True, use_interaction=True):
        super(GraphLevelAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.use_hidden_rel = use_hidden_rel
        self.use_interaction = use_interaction

        if self.use_hidden_rel:
            self.max_len = 3
        else:
            self.max_len = 2

        # Shared encoder
        self.shared_encoder = nn.Linear(self.max_len * input_dim, embed_dim)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        # Linear transformations for attention score computation for each embedding type
        self.attention_hyper = nn.Linear(3*embed_dim, 1)
        self.attention_hetero = nn.Linear(3*embed_dim, 1)
        if use_interaction:
            self.attention_inter = nn.Linear(3*embed_dim, 1)

        nn.init.xavier_uniform_(self.attention_hyper.weight)
        nn.init.xavier_uniform_(self.attention_hetero.weight)
        if use_interaction:
            nn.init.xavier_uniform_(self.attention_inter.weight)

        # Gating mechanism for interaction embedding
        if use_interaction:
            self.gate_weight = nn.Linear(embed_dim, embed_dim)
            self.gate_bias = nn.Parameter(torch.zeros(embed_dim))

        # Non-linear activation function
        self.activation = nn.GELU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.20)
        self.bn = nn.BatchNorm1d(embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hetero_list, hyper_list, inter_emb=None):
        hetero_padded = self._pad_list(hetero_list, self.max_len)
        hyper_padded = self._pad_list(hyper_list, self.max_len)

        hetero_concat = torch.cat(hetero_padded, dim=-1)
        hyper_concat = torch.cat(hyper_padded, dim=-1)

        hetero_encoded = self.shared_encoder(hetero_concat)
        hyper_encoded = self.shared_encoder(hyper_concat)
        hetero_encoded = self.bn(hetero_encoded)
        hyper_encoded = self.bn(hyper_encoded)

        hetero_encoded = self.activation(hetero_encoded)
        hyper_encoded = self.activation(hyper_encoded)

        if self.use_interaction:
            inter_emb = self.activation(inter_emb)

            # Gating mechanism for interaction embedding
            gate = torch.tanh(self.gate_weight(inter_emb) + self.gate_bias)
            inter_emb = inter_emb * gate  # Element-wise multiplication
        else:
            inter_emb = torch.zeros_like(hyper_encoded)

        # Step 1: Combine the embeddings
        combined_scores = torch.cat([hyper_encoded, hetero_encoded, inter_emb], dim=-1)  # [N, 3*embed_dim]

        # Step 2: Compute attention scores using separate linear layers
        r_hyper = self.attention_hyper(combined_scores)  # [N, 1]
        r_hetero = self.attention_hetero(combined_scores)  # [N, 1]
        if self.use_interaction:
            r_inter = self.attention_inter(combined_scores)  # [N, 1]

        # Step 3: Combine attention scores and apply softmax
        if self.use_interaction:
            combined_scores = torch.cat([r_hyper, r_hetero, r_inter], dim=-1)  # [N, 3]
            attention_weights = F.softmax(combined_scores, dim=-1)  # [N, 3]
        else:
            combined_scores = torch.cat([r_hyper, r_hetero], dim=-1)  # [N, 2]
            attention_weights = F.softmax(combined_scores, dim=-1)  # [N, 2]
            attention_weights = F.pad(attention_weights, (0, 1), value=0)  # Pad with zeros for inter if not used
        #print(attention_weights[0, 0],attention_weights[0, 1],attention_weights[0, 2])
        # Step 4: Compute the final fused embedding
        final_emb = (attention_weights[:, 0].unsqueeze(1) * hyper_encoded) + \
                    (attention_weights[:, 1].unsqueeze(1) * hetero_encoded) + \
                    (attention_weights[:, 2].unsqueeze(1) * inter_emb)

        return final_emb

    def _pad_list(self, feature_list, max_len):
        N, F = feature_list[0].size()
        padded_list = feature_list[:]
        if len(feature_list) < max_len:
            for _ in range(max_len - len(feature_list)):
                padded_list.append(torch.zeros((N, F), device=feature_list[0].device))

        return padded_list

'''
class GraphLevelAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout_rate=0.5, use_hidden_rel=True ,use_interaction=True):
        super(GraphLevelAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.use_hidden_rel = use_hidden_rel
        self.use_interaction = use_interaction  # 控制是否使用交互项
        if self.use_hidden_rel:     #其实不用担心如果成对关系没有推断出来的，因为即使没有推断出来也会进行补齐
            self.max_len = 3
        else:
            self.max_len = 2
        # Shared linear transformation (encoder)
        self.shared_encoder = nn.Linear(self.max_len * input_dim, embed_dim)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        # Linear transformations for attention score computation
        self.attention_hyper = nn.Linear(embed_dim, 1)
        self.attention_hetero = nn.Linear(embed_dim, 1)
        if use_interaction:
            self.attention_inter = nn.Linear(embed_dim, 1)
            nn.init.xavier_uniform_(self.attention_inter.weight)

        nn.init.xavier_uniform_(self.attention_hyper.weight)
        nn.init.xavier_uniform_(self.attention_hetero.weight)

        # Gating mechanism for interaction embedding
        if use_interaction:
            self.gate_weight = nn.Linear(embed_dim, embed_dim)
            self.gate_bias = nn.Parameter(torch.zeros(embed_dim))

        # Non-linear activation function
        self.activation = nn.GELU()
        self.leaky_relu= nn.LeakyReLU(negative_slope=0.01)
        self.bn = nn.BatchNorm1d(embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hetero_list, hyper_list, inter_emb=None):
        hetero_padded = self._pad_list(hetero_list, self.max_len)
        hyper_padded = self._pad_list(hyper_list, self.max_len)

        hetero_concat = torch.cat(hetero_padded, dim=-1)
        hyper_concat = torch.cat(hyper_padded, dim=-1)

        hetero_encoded = self.shared_encoder(hetero_concat)
        hyper_encoded = self.shared_encoder(hyper_concat)
        hetero_encoded = self.bn(hetero_encoded)
        hyper_encoded = self.bn(hyper_encoded)

        hetero_encoded = self.activation(hetero_encoded)
        hyper_encoded = self.activation(hyper_encoded)

        if self.use_interaction:
            inter_emb = self.activation(inter_emb)

            # Gating mechanism for interaction embedding
            gate = torch.tanh(self.gate_weight(inter_emb) + self.gate_bias)
            inter_emb = inter_emb * gate  # Element-wise multiplication

            r_inter = self.attention_inter(inter_emb)  # 计算交互项的注意力得分
        else:
            inter_emb = torch.zeros_like(hyper_encoded)  # 不使用交互项时，设置为零

        r_hyper = self.attention_hyper(hyper_encoded)
        r_hetero = self.attention_hetero(hetero_encoded)

        exp_r_hyper = torch.exp(r_hyper)
        exp_r_hetero = torch.exp(r_hetero)

        if self.use_interaction:
            exp_r_inter = torch.exp(r_inter)
            sum_exp = exp_r_hyper + exp_r_hetero + exp_r_inter

            omega_inter = exp_r_inter / sum_exp
        else:
            sum_exp = exp_r_hyper + exp_r_hetero
            omega_inter = torch.zeros_like(exp_r_hyper)  # 交互项权重为零

        omega_hyper = exp_r_hyper / sum_exp
        omega_hetero = exp_r_hetero / sum_exp

        # Compute the final weighted embedding
        final_emb = (omega_hyper * hyper_encoded) + (omega_hetero * hetero_encoded) + (omega_inter * inter_emb)
        #print(omega_hyper[0],omega_hetero[0],omega_inter[0])

        return final_emb

    def _pad_list(self, feature_list, max_len):
        N, F = feature_list[0].size()
        padded_list = feature_list[:]
        if len(feature_list) < max_len:
            for _ in range(max_len - len(feature_list)):
                padded_list.append(torch.zeros((N, F), device=feature_list[0].device))

        return padded_list
'''

'''
#这个缺少门控机制
import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared Encoder for Cross-Graph Feature Mapping and Attention-Based Feature Fusion
class GraphLevelAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout_rate=0.5):
        """
        Unified feature fusion module via attention mechanism.
        :param input_dim: Dimension of the input node features
        :param embed_dim: Dimension of the output embedding after transformation
        :param max_len: Maximum length of the heterogeneous or hypergraph list after padding
        :param dropout_rate: Dropout rate for regularization
        """
        super(GraphLevelAttention, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = 2
        self.dropout_rate = dropout_rate

        # Shared linear transformation (encoder) to map input into a unified space
        self.shared_encoder = nn.Linear(self.max_len * input_dim, embed_dim)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        # Linear transformations for attention score computation for hypergraph, heterogeneous graph, and interaction
        self.attention_hyper = nn.Linear(embed_dim, 1)
        self.attention_hetero = nn.Linear(embed_dim, 1)
        self.attention_inter = nn.Linear(embed_dim, 1)
        nn.init.xavier_uniform_(self.attention_hyper.weight)
        nn.init.xavier_uniform_(self.attention_hetero.weight)
        nn.init.xavier_uniform_(self.attention_inter.weight)

        # Non-linear activation function
        self.activation = nn.GELU()
        self.bn = nn.BatchNorm1d(embed_dim)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hetero_list, hyper_list, inter_emb):
        """
        Forward pass for unified feature fusion.
        :param hetero_list: List of tensors representing node features from heterogeneous graph, each [N, F]
        :param hyper_list: List of tensors representing node features from hypergraph, each [N, F]
        :param inter_emb: The interaction embedding between hypergraph and heterogeneous graph, [N, F]
        :return: Final fused node embeddings, [N, F]
        """

        # Step 1: Padding the heterogeneous and hypergraph lists to ensure same length (max_len)
        hetero_padded = self._pad_list(hetero_list, self.max_len)
        hyper_padded = self._pad_list(hyper_list, self.max_len)

        # Step 2: Concatenating all node features in the list
        hetero_concat = torch.cat(hetero_padded, dim=-1)  # [N, L_max * F]
        hyper_concat = torch.cat(hyper_padded, dim=-1)  # [N, L_max * F]

        # Step 3: Apply the shared encoder to map concatenated features into the same space
        hetero_encoded = self.shared_encoder(hetero_concat)  # [N, embed_dim]
        hyper_encoded = self.shared_encoder(hyper_concat)  # [N, embed_dim]
        hetero_encoded = self.bn(hetero_encoded)
        hyper_encoded = self.bn(hyper_encoded)
        # Step 4: Non-linear transformation for encoded features
        hetero_encoded = self.activation(hetero_encoded)
        hyper_encoded = self.activation(hyper_encoded)
        inter_emb = self.activation(inter_emb)  # [N, F], already in the same space

        # Step 5: Compute attention scores for hypergraph, heterogeneous graph, and interaction term
        r_hyper = self.attention_hyper(hyper_encoded)  # [N, 1]
        r_hetero = self.attention_hetero(hetero_encoded)  # [N, 1]
        r_inter = self.attention_inter(inter_emb)  # [N, 1]

        # Step 6: Apply softmax to compute attention weights
        exp_r_hyper = torch.exp(r_hyper)
        exp_r_hetero = torch.exp(r_hetero)
        exp_r_inter = torch.exp(r_inter)

        sum_exp = exp_r_hyper + exp_r_hetero + exp_r_inter
        omega_hyper = exp_r_hyper / sum_exp  # [N, 1]
        omega_hetero = exp_r_hetero / sum_exp  # [N, 1]
        omega_inter = exp_r_inter / sum_exp  # [N, 1]

        # Step 7: Compute the final weighted embedding by summing the contributions
        final_emb = (omega_hyper * hyper_encoded) + (omega_hetero * hetero_encoded) + (omega_inter * inter_emb)  # [N, F]
        #print('hyper',omega_hyper[0],'heter',omega_hetero[0],'inter',omega_inter[0])

        # Step 8: Apply dropout for regularization
        #final_emb = self.dropout(final_emb)

        return final_emb
    def _pad_list(self, feature_list, max_len):
        """
        Padding function to pad the list of features to the same length.
        :param feature_list: List of [N, F] tensors (heterogeneous or hypergraph node features)
        :param max_len: Target maximum length for padding
        :return: Padded list of tensors with [N, max_len * F]
        """
        N, F = feature_list[0].size()
        padded_list = feature_list[:]
        if len(feature_list) < max_len:
            for _ in range(max_len - len(feature_list)):
                # Pad with zeros if list is shorter than max_len
                padded_list.append(torch.zeros((N, F), device=feature_list[0].device))

        return padded_list'''
