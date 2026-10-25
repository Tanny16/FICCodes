# encoding: utf-8

from utils import gen_graph_pairs
from models import GraphSeqEmbeddingModel
from syn_data import GraphSeqGenerator
import matplotlib.pyplot as plt
import numpy as np

graphs_generator = GraphSeqGenerator()
subgraphs_wrt_staff, seq_data, labels = graphs_generator.gen_subgraph_seq()
embedding_model = GraphSeqEmbeddingModel()

graph_seq = []
contrastive_pairs = []
for graph in subgraphs_wrt_staff[0].values():
    graph_seq.append(graph)
    contrastive_pairs.append(gen_graph_pairs(graph))
embedding_model.train(graph_seq, contrastive_pairs)

graph_embeddings = embedding_model.graphs_embed(graph_seq).detach().numpy()

pos_index = np.array(labels[0]) == 1
neg_index = np.array(labels[0]) == 0

plt.scatter(graph_embeddings[pos_index, 0], graph_embeddings[pos_index, 1], color='red', marker='*', label='abnormal')
plt.scatter(graph_embeddings[neg_index, 0], graph_embeddings[neg_index, 1], color='blue', marker='.', label='normal')
plt.legend(loc='best')
