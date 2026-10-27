''' Define the HGTAN model '''
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
#from HGTAN.tri_attn import HGAT
#from HGTAN.temp_layers import TemporalAttention
import torch
import torch.nn as nn
from training.tools import *
import torch.nn.functional as F
from MSDGNN.layers import *
import torch.fft


class MSDGNN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            num_stock, rnn_unit, n_hid, n_class,
            feature,
            d_word_vec, d_model, dropout,
            tgt_emb_prj_weight_sharing,scale_num,path_num,window_size,mem_dim):

        super().__init__()
        self.dropout = dropout
        self.linear = nn.Linear(feature, d_word_vec)
        self.scale_num = scale_num
        self.path_num = path_num
        self.n_hid=n_hid
        self.window_size = window_size
        # torch.nn.init.xavier_uniform_(self.linear.weight)
        # if self.linear.bias is not None:
        #     torch.nn.init.zeros_(self.linear.bias)


        # self.scgrn = SCGRN(
        #     input_dim=rnn_unit,
        #     scale_num=scale_num,
        #     path_num=path_num,
        #     n_hid=n_hid
        # )



        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)

        self.linear = nn.Linear(scale_num*rnn_unit, n_hid, bias=False)
        #self.tgt_word_prj = nn.Linear(2*rnn_unit, n_class, bias=False)
        #self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        self.rnn2 = nn.GRU(feature,
                           rnn_unit,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
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

        self.MDS = MarketDrivenSpectralAttention(T=self.window_size, N=num_stock, C=feature, embed_dim=n_hid)
        self.AFR = AdaptiveFrequencyRelationInference(T=self.window_size, N=num_stock, C=feature, K=scale_num, embed_dim=n_hid)
        self.PR = ProgressiveRiskPropagation(T=self.window_size,N=num_stock, K=scale_num, F=n_hid, S=3, kernel_sizes=[3, 5, 7])
        # self.phase_net = PhaseNetwork(T=self.window_size, C=feature, D=mem_dim, Theta=60)
        # #self.amplitude_net = AmplitudeNetwork(T=self.window_size, C=feature, F=n_hid,D=mem_dim, Theta=60, kernel_size=1, norm="layer",groups=2, eps =1e-6)
        # self.amplitude_net = AmplitudeNetwork(T=self.window_size, C=feature,D=mem_dim, Theta=60, eps =1e-6)
        # self.reconstruction_net = ReconstructionNetwork(T=self.window_size, K=self.window_size // 2 + 1, feature_dim = n_hid)
        #
        # self.knowledge = KnowledgeGuidedFrequencyAlignment(T=self.window_size, C=feature, P=60)
        #
        # #self.relation_net = DynamicRelationExtraction(K = self.window_size // 2 + 1, top_k = 5, tau_factor = 1)
        # #self.ProgressiveRiskEmbedding = ProgressiveRiskEmbedding(T=self.window_size, K = self.window_size // 2 + 1, num_sub_bands = 3, embed_dim = n_hid, top_k = 5)
        # # Initialize the new module
        # #self.feature_extractor = TemporalFeatureExtractor(T=self.window_size, K=5, feature_dim=n_hid)
        # self.routing_module = StateAwareGraphRouting(input_dim= self.window_size // 2 + 1, num_modes=5, d_k= n_hid, tau=0.5)
        # self.dre_module = DynamicRelationExtraction(C=5, tau_factor=1)

        self.ln = nn.LayerNorm(n_hid)
        self.ln2 = nn.LayerNorm(rnn_unit * scale_num)
        self.ln_in = nn.LayerNorm(feature)
        self.ln3 = nn.LayerNorm(2*n_hid)
        self.ln5 = nn.LayerNorm(n_hid)
        #self.ln2 = nn.LayerNorm(rnn_unit * scale_num)

        #self.mstdm = MultiScaleTemporalDecompositionModule(T=T, C=C, K=3, output_feature_dim=64)
    def forward(self,src_seq):
        stock_num = src_seq.size(1)
        seq_len = src_seq.size(0)


        if torch.isnan(src_seq).any():
            print("src_seq1 中存在 NaN 值！")


        #xinfre = self.knowledge(src_seq)

        h_n = self.MDS(src_seq)

        R, h_n2 = self.AFR(h_n)

        h_n3 = self.ln(self.PR(h_n2,R))

        h_n3 = F.dropout(h_n3, self.dropout, training=self.training)




        #combined_output = torch.cat((h_n,temporal_features), dim=1)

        #combined_output = F.dropout(combined_output, self.dropout, training=self.training)
        #combined_output = s_n
        seq_logit = F.elu(self.tgt_word_prj(h_n3))
        #
        output = F.log_softmax(seq_logit, dim=1)
        # y_tem = F.log_softmax(output, dim=1)
        # output: torch.Size([385, 2])



        return output


