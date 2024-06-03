import random
import numpy as np
import torch
from tqdm import tqdm
import pickle
import copy


def all_node_negative(data, node_type_list, num_nodes, split_type):

    def node_negative_sampling(select_node_pool_src, select_node_pool_dst):
        '''
        需要确保all edge 是可以识别的格式
        '''

        node_src_with_matidx = dict(zip(select_node_pool_src, range(len(select_node_pool_src))))
        node_dst_with_matidx = dict(zip(select_node_pool_dst, range(len(select_node_pool_dst))))

        rev_node_src_with_matidx = dict(zip(range(len(select_node_pool_src)), select_node_pool_src))
        # rev_node_dst_with_matidx = dict(zip(range(len(select_node_pool_dst)), select_node_pool_dst))

        mat_adj = np.zeros((len(select_node_pool_src), len(select_node_pool_dst)))

        row, col = data.edge_index
        edge_type_list = data.edge_type

        mask = row < col
        mask_type = torch.tensor([i == split_type for i in edge_type_list])

        resulting_mask = mask & mask_type
        row, col = row[resulting_mask], col[resulting_mask]
        row = row.cpu().numpy().tolist()
        col = col.cpu().numpy().tolist()

        for r, c in tqdm(zip(row, col), total=len(row)):
            if r in select_node_pool_src:
                mat_adj[node_src_with_matidx[r], node_dst_with_matidx[c]] = 1
                # mat_adj[node_dst_with_matidx[c], node_src_with_matidx[r]] = 1
            else:
                # r in select_node_pool_src:
                mat_adj[node_src_with_matidx[c], node_dst_with_matidx[r]] = 1
                # mat_adj[node_dst_with_matidx[r], node_src_with_matidx[c]] = 1

        negative_samples_by_node = {}
        num_negative_by_node = {}
        for row_ in tqdm(range(len(mat_adj))):
            # Get indices where the value is 0
            row = mat_adj[row_]
            negative_indices = select_node_pool_dst[np.where(row == 0)[0]]
            # Append these indices to the list

            maprow = rev_node_src_with_matidx[row_]
            negative_samples_by_node[maprow] = negative_indices
            num_negative_by_node[maprow] = len(negative_indices)
        return negative_samples_by_node, num_negative_by_node

    def type_neg_samping(split_type: int, node_type_list: list, num_nodes, all_edge=None):
        '''
        pos--> no need for pos edges
        all_edge--> edge_index format, avoid involving test/val edges as the neg edges of the training data.
        '''
        print('Calling type_neg_sampling to sample on a certain edge type')
        # sample based on the original edge ordering
        # all_edge = all_edge.t().tolist()
        node_type_list = node_type_list.tolist()
        assert split_type == 2
        srcNode = 0
        dstNode = 3

        '''
        使用node_type_list, 不需要确保顺序？这里有问题，
        虽然目前的predict_dataset做的没错，node和edge都固定了，但要注意node_type_list是不是总可以保持顺序，
        使用node_type_mapping更好
        '''
        print(node_type_list)
        get_src_node = [i[0] == srcNode for i in node_type_list]
        get_dst_node = [i[0] == dstNode for i in node_type_list]

        node_list = np.arange(num_nodes)
        src_node = node_list[get_src_node]
        dst_node = node_list[get_dst_node]

        negative_edges, negative_counts = node_negative_sampling(src_node, dst_node)
        # print(type(negative_edges), len(negative_edges), negative_edges[:10], negative_edges[-10:], negative_edges[100000:100010])
        return negative_edges, negative_counts

    neg_edge_index, negative_counts = type_neg_samping(split_type=split_type, node_type_list=node_type_list, num_nodes=num_nodes)
    return neg_edge_index, negative_counts

#
# from predlink_dataset import LinkPredictionDataset
# from torch.utils.data import DataLoader
# nxgraph = pickle.load(open('../pkl/sub20_graph_phe.pkl', 'rb'))
# graph_dataset = LinkPredictionDataset(nxgraph)
# data = graph_dataset.data
# # def all_node_negative(data, node_type_list, num_nodes, split_type, neg_num):
# node_type_list = graph_dataset.data.node_type
# neg_edges, neg_counts = all_node_negative(data, node_type_list, data.num_nodes, split_type=2)
# print(neg_edges, neg_counts)
# with open('neg_edges_phe.pkl', 'wb') as f:
#     pickle.dump(neg_edges, f)
# with open('neg_counts_phe.pkl', 'wb') as f:
#     pickle.dump(neg_counts, f)

def get_negative_for_pos_edges(data, split_edge ):
    def get_source_nodes(data, srcNode=0):
        node_type_list = data.node_type
        node_type_list = node_type_list.tolist()

        # print(len(node_type_list))
        num_nodes = len(node_type_list)
        get_src_node = [i[0] == srcNode for i in node_type_list]
        node_list = np.arange(num_nodes)
        src_nodes = node_list[get_src_node]
        return src_nodes

    def get_pos_edge_eids(pos_edges, src_nodes):

        pos_edges = copy.deepcopy(pos_edges)
        pos_edges1, pos_edges2 = pos_edges[:, 0].cpu().numpy(), pos_edges[:, 1].cpu().numpy()
        pos_edges_index = [0 if i in src_nodes else 1 for i in tqdm(pos_edges1)]

        pos_edges_index = np.array(pos_edges_index)
        pos_edges = pos_edges.numpy()

        pos_edges_eid = pos_edges[np.arange(pos_edges_index.shape[0]), pos_edges_index]
        return pos_edges_eid

    def per_eid_neg(pos_eid, train_neg_dict, neg_number):
        sampled_negs_for_train = []
        for i in tqdm(pos_eid):
            sample_ = random.sample(list(train_neg_dict[i]), neg_number)
            sampled_negs_for_train.append(sample_)
        sampled_negs_for_train = np.array(sampled_negs_for_train)
        return sampled_negs_for_train

    def read_all_neg_edges(split=(300, 200, 300)):
        split_dicts = [{}, {}, {}]
        neg_edges = pickle.load(open('neg_edges_phe.pkl', 'rb'))
        neg_counts = pickle.load(open('neg_counts_phe.pkl', 'rb'))
        min_neg = min(neg_counts.values())

        for key, value in neg_edges.items():
            permuted_values = random.sample(list(value), len(value))
            split_dicts[0][key] = permuted_values[:split[0]]
            split_dicts[1][key] = permuted_values[split[0]:split[0] + split[1]]
            split_dicts[2][key] = permuted_values[split[0] + split[1]: split[1] + split[2] + split[0]]
        print('read all neg counts, find min neg: ', min_neg)
        return split_dicts
        # test_neg: 300
        # val_neg: 200
        # train_neg: 300

    eid_nodes = get_source_nodes(data, srcNode=0)
    neg_edges_train, neg_edges_val, neg_edges_test = read_all_neg_edges((300, 300, 300))  # read only train

    print('Get negative per training pos edges')
    pos_edge_eids_train = get_pos_edge_eids(split_edge['train']['edge'], eid_nodes)
    # print(pos_edge_eids_train, pos_edge_eids_train[-5:], split_edge['train']['edge'][-5:, 0], split_edge['train']['edge'][-5:, 1])
    sampled_negs_for_train = per_eid_neg(pos_edge_eids_train, neg_edges_train, 300)

    print('Get negative per valid pos edges')

    pos_edge_eids_val = get_pos_edge_eids(split_edge['valid']['edge'], eid_nodes)
    sampled_negs_for_val = per_eid_neg(pos_edge_eids_val, neg_edges_val, 300)

    print('Get negative per test pos edges')

    pos_edge_eids_test = get_pos_edge_eids(split_edge['test']['edge'], eid_nodes)
    sampled_negs_for_test = per_eid_neg(pos_edge_eids_test, neg_edges_test, 300)

    # sample numpy:
    device = split_edge['valid']['edge'].device
    split_edge['train']['eid_edge_neg'] = torch.tensor(sampled_negs_for_train).to(device)
    split_edge['train']['edge_eid'] = torch.tensor(pos_edge_eids_train).to(device)
    split_edge['valid']['eid_edge_neg'] = torch.tensor(sampled_negs_for_val).to(device)
    split_edge['valid']['edge_eid'] = torch.tensor(pos_edge_eids_val).to(device)
    split_edge['test']['eid_edge_neg'] = torch.tensor(sampled_negs_for_test).to(device)
    split_edge['test']['edge_eid'] = torch.tensor(pos_edge_eids_test).to(device)

    return split_edge

'''
from predlink_dataset import LinkPredictionDataset
from torch.utils.data import DataLoader

nxgraph = pickle.load(open('../pkl/sub20_graph_phe.pkl', 'rb')) #
graph_dataset = LinkPredictionDataset(nxgraph)
data = graph_dataset.data
# split_edge = torch.load("../pkl/" + 'ehr' + "_{}_{}_seed{}.pkl".format('nosplit_code_ed', '2', str(0)))
# split_edge = get_negative_for_pos_edges(data, split_edge )

# pos_valid_edge = split_edge['valid']['edge']
# pos_src = split_edge['valid']['edge_eid']
# neg_edges = split_edge['valid']['eid_edge_neg']
# for i, perm in enumerate(tqdm(DataLoader(range(pos_valid_edge.size(0)), 128), total=len(DataLoader(range(pos_valid_edge.size(0)), 128)))):
#     edge = pos_valid_edge[perm]
#     print(type(perm), type(edge))
#     edge_eid = pos_src[perm]
#     edge_neg = neg_edges[perm]
#     negs = []
#     for e, neg_row in zip(edge_eid, edge_neg):
#         col_index = torch.randint(0, neg_row.size(0), (1,))  #  Randomly choose one index.
#         negs.append((e , neg_row[col_index] ))  #
#     negs = torch.tensor(negs)
#
#     print(edge.shape, negs.shape, negs)
#
neg_counts = pickle.load(open('neg_edges_phe.pkl', 'rb'))
diag = []
def get_source_nodes(data, srcNode=0):
    node_type_list = data.node_type
    node_type_list = node_type_list.tolist()

    # print(len(node_type_list))
    num_nodes = len(node_type_list)
    get_src_node = [i[0] == srcNode for i in node_type_list]
    node_list = np.arange(num_nodes)
    src_nodes = node_list[get_src_node]
    return src_nodes


eid_nodes = get_source_nodes(data, srcNode=3)
from collections import Counter

for i , j in neg_counts.items():
    idiag = [di for di in eid_nodes if di not in j]
    print(len(idiag))
    diag.extend(idiag)
diag_c = Counter(diag)
print(diag_c)
# 15339
print(sorted([1108 - i for i in list(neg_counts.values())]))
'''
