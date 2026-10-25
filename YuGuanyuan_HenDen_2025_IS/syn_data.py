# encoding: utf-8

from dgl import heterograph, DGLError
import torch.nn.functional as F
import torch
import numpy as np
from random import shuffle, randint
from collections import defaultdict
from settings import MetaRelations


class GraphSeqGenerator(object):
    def __init__(self):
        self.OPE_RE = MetaRelations.OPE_RE
        self.OPE_RE_ = MetaRelations.OPE_RE_
        self.DEL_RE = MetaRelations.DEL_RE
        self.DEL_RE_ = MetaRelations.DEL_RE_
        self.ADD_RE = MetaRelations.ADD_RE
        self.ADD_RE_ = MetaRelations.ADD_RE_
        self.UPD_RE = MetaRelations.UPD_RE
        self.UPD_RE_ = MetaRelations.UPD_RE_
        self.QUE_RE = MetaRelations.QUE_RE
        self.QUE_RE_ = MetaRelations.QUE_RE_
        self.DOW_RE = MetaRelations.DOW_RE
        self.DOW_RE_ = MetaRelations.DOW_RE_

        self.ntype2id = {'staff': 0, 'address': 1, 'system': 2}
        self.etype2id = {'operate': 0, 'operate_': 1, 'delete': 2, 'delete_': 3,
                         'add': 4, 'add_': 5, 'update': 6, 'update_': 7, 'query': 8,
                         'query_': 9, 'download': 10, 'download_': 11}

        self.top_op = [(op, sys) for op in (self.DEL_RE, self.ADD_RE) for sys in range(2)]  # 4
        self.mid_op = [(self.UPD_RE, sys) for sys in range(2, 5)]  # 3
        self.bot_op = [(op, sys) for op in (self.QUE_RE, self.DOW_RE) for sys in range(5, 10)]  # 10

        self.num_staff = 1000
        self.num_day = 100

    @staticmethod
    def _gen_staff():
        # 共1000个员工：500个行内，500个外包
        label = torch.arange(2).repeat_interleave(repeats=torch.LongTensor([500, 500]))
        authority = torch.arange(3).repeat_interleave(repeats=torch.LongTensor([200, 300, 500]))
        return torch.cat([F.one_hot(label, num_classes=2).to(torch.float32),
                          F.one_hot(authority, num_classes=3).to(torch.float32)], dim=-1)

    @staticmethod
    def _gen_address():
        # 共1000个IP地址：800个行内，200个行外
        label = torch.arange(2).repeat_interleave(repeats=torch.LongTensor([800, 200]))
        return F.one_hot(label, num_classes=2).to(torch.float32)

    @staticmethod
    def _gen_system():
        # 共10个系统：2个高级权限，3个中级权限，5个低级权限
        label = torch.arange(3).repeat_interleave(repeats=torch.LongTensor([2, 3, 5]))
        return F.one_hot(label, num_classes=3).to(torch.float32)

    def _gen_init_data(self):
        staff_set = np.arange(1000, dtype=np.int32)
        address_set = np.random.permutation(np.arange(1000, dtype=np.int32))

        init_dataset = []

        all_op = self.top_op + self.mid_op + self.bot_op
        shuffle(all_op)
        # 构建初始化图
        # 权限最高的前200名员工，拥有delete, add, update, query, download等操作权限
        for (staff_id, address) in zip(staff_set[:200], address_set[:200]):
            init_data = {self.OPE_RE: ([staff_id], [address])}
            init_data.update({op: ([address], [sys]) for op, sys in all_op[:randint(0, 17)]})
            init_dataset.append(init_data)

        # 中等权限的300名员工，拥有update, query, download等权限
        all_op = self.mid_op + self.bot_op
        shuffle(all_op)
        for (staff_id, address) in zip(staff_set[200:500], address_set[200:500]):
            init_data = {self.OPE_RE: ([staff_id], [address])}
            init_data.update({op: ([address], [sys]) for op, sys in all_op[:randint(0, 13)]})
            init_dataset.append(init_data)

        # 权限最低的500名员工，拥有query, download等操作权限
        all_op = self.bot_op
        shuffle(all_op)
        for (staff_id, address) in zip(staff_set[500:], address_set[500:]):
            init_data = {self.OPE_RE: ([staff_id], [address])}
            init_data.update({op: ([address], [sys]) for op, sys in all_op[:randint(0, 10)]})
            init_dataset.append(init_data)

        return init_dataset

    def _gen_seq_data(self, init_dataset):
        seq_data_for_staff = []
        labels = []
        # 生成100天的后续数据
        # 前250名员工，账号共用
        for staff_data in init_dataset[:250]:
            seq_data = [staff_data.copy() for _ in range(self.num_day)]
            seq_labels = [0] * self.num_day
            for date in np.random.permutation(np.arange(self.num_day, dtype=np.int32))[:20]:
                seq_labels[date] = 1

                src, dst = seq_data[date][self.OPE_RE]
                num_change = randint(1, 5)
                aug_src = src * num_change
                aug_dst = np.random.permutation(np.arange(1000, dtype=np.int32))[:(num_change-1)]
                aug_dst = aug_dst.tolist() + dst
                seq_data[date][self.OPE_RE] = (aug_src, aug_dst)

            seq_data_for_staff.append(seq_data)
            labels.append(seq_labels)

        # 后750名员工，越权使用系统
        for staff_data in init_dataset[250:]:
            seq_data = [staff_data.copy() for _ in range(self.num_day)]
            seq_labels = [0] * self.num_day
            all_op = self.top_op
            shuffle(all_op)
            for date in np.random.permutation(np.arange(self.num_day, dtype=np.int32))[:20]:
                seq_labels[date] = 1
                seq_data[date].update({op: ([seq_data[date][self.OPE_RE][1][0]], [sys])
                                       for op, sys in all_op[:randint(0, 4)]})
            seq_data_for_staff.append(seq_data)
            labels.append(seq_labels)

        return seq_data_for_staff, labels

    def _trans2tensor(self, specific_data):
        tensor_dict = {}
        for op in specific_data:
            src, dst = specific_data[op]
            src = torch.tensor(src, dtype=torch.int32)
            dst = torch.tensor(dst, dtype=torch.int32)
            tensor_dict[op] = (src, dst)
            if op == self.OPE_RE:
                tensor_dict[self.OPE_RE_] = (dst, src)
            if op == self.DEL_RE:
                tensor_dict[self.DEL_RE_] = (dst, src)
            if op == self.ADD_RE:
                tensor_dict[self.ADD_RE_] = (dst, src)
            if op == self.UPD_RE:
                tensor_dict[self.UPD_RE_] = (dst, src)
            if op == self.QUE_RE:
                tensor_dict[self.QUE_RE_] = (dst, src)
            if op == self.DOW_RE:
                tensor_dict[self.DOW_RE_] = (dst, src)
        return tensor_dict

    def gen_complete(self):
        init_dataset = self._gen_init_data()
        seq_data, labels = self._gen_seq_data(init_dataset)

        hg_for_days = {}
        for day in range(self.num_day):
            graph_data = {}
            for staff in range(self.num_staff):
                specific_data = seq_data[staff][day]
                for op in specific_data:
                    src_in_graph, dst_in_graph = graph_data.get(op, ([], []))
                    src, dst = specific_data[op]
                    src_in_graph += src
                    dst_in_graph += dst
                    graph_data[op] = (src_in_graph, dst_in_graph)

            graph_data = self._trans2tensor(graph_data)
            hg = heterograph(graph_data)
            hg.nodes['staff'].data['x'] = self._gen_staff()
            hg.nodes['address'].data['x'] = self._gen_address()
            hg.nodes['system'].data['x'] = self._gen_system()
            hg_for_days[day] = hg
        return hg_for_days, torch.IntTensor(labels)

    @staticmethod
    def remap_node_id(data_dict):
        node_id_mapper = defaultdict(dict)
        for meta_rel in data_dict:
            src_type, _, dst_type = meta_rel
            src_ids, dst_ids = data_dict[meta_rel]
            for src_id, dst_id in zip(src_ids, dst_ids):
                if src_id not in node_id_mapper[src_type]:
                    node_id_mapper[src_type].update({src_id: len(node_id_mapper[src_type])})
                if dst_id not in node_id_mapper[dst_type]:
                    node_id_mapper[dst_type].update({dst_id: len(node_id_mapper[dst_type])})
        new_data_dict = {}
        for meta_rel in data_dict:
            src_type, _, dst_type = meta_rel
            src_ids, dst_ids = data_dict[meta_rel]
            new_src_ids = []
            new_dst_ids = []
            for src_id, dst_id in zip(src_ids, dst_ids):
                new_src_id = node_id_mapper[src_type][src_id]
                new_dst_id = node_id_mapper[dst_type][dst_id]
                new_src_ids.append(new_src_id)
                new_dst_ids.append(new_dst_id)

            new_data_dict[meta_rel] = (new_src_ids, new_dst_ids)
        return new_data_dict, node_id_mapper

    def gen_subgraph_seq(self):
        init_dataset = self._gen_init_data()
        seq_data, labels = self._gen_seq_data(init_dataset)
        staff_feats = self._gen_staff()
        address_feats = self._gen_address()
        system_feats = self._gen_system()

        subgraphs_wrt_staff = {}
        for staff in range(self.num_staff):
            subgraphs_wrt_date = {}
            for day in range(self.num_day):
                specific_data = seq_data[staff][day]
                specific_data, node_id_mapper = self.remap_node_id(specific_data)
                graph_data = self._trans2tensor(specific_data)
                hg = heterograph(graph_data)
                try:
                    hg.nodes['staff'].data['x'] = staff_feats[list(node_id_mapper['staff'].keys()), :]
                except DGLError as e:
                    print(f'{e}, staff: {staff}, day: {day}')

                try:
                    hg.nodes['address'].data['x'] = address_feats[list(node_id_mapper['address'].keys()), :]
                except DGLError as e:
                    print(f'{e}, staff: {staff}, day: {day}')

                try:
                    hg.nodes['system'].data['x'] = system_feats[list(node_id_mapper['system'].keys()), :]
                except DGLError as e:
                    print(f'{e}, staff: {staff}, day: {day}')

                hg.ntype2id = self.ntype2id
                for etype in hg.etypes:
                    hg.edges[etype].data['etype_id'] = torch.ones(hg.number_of_edges(etype),
                                                                  dtype=torch.long) * self.etype2id[etype]

                subgraphs_wrt_date[day] = hg
            subgraphs_wrt_staff[staff] = subgraphs_wrt_date
        return subgraphs_wrt_staff, seq_data, labels

