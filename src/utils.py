
import pickle
import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from torch_sparse import spspmm
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
import pdb
import math

import torch

import torch_geometric
from torch_geometric.deprecation import deprecated
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
import tqdm
import copy
from per_edge_negative import get_negative_for_pos_edges


def neighbors(fringe, A, outgoing=True):
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl'):
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists)==0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z>100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                node_id=node_ids, num_nodes=num_nodes)
    return data


def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl',
                                ratio_per_hop=1.0, max_nodes_per_hop=None,
                                directed=False, A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                             max_nodes_per_hop, node_features=x, y=y,
                             directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col  # mask all the values
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]  # permute the edge orders
        r, c = row[:n_v], col[:n_v]  # select for validation
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]  # select for test
        data.test_pos_edge_index = torch.stack([r, c], dim=0) # stack for test edge index
        r, c = row[n_v + n_t:], col[n_v + n_t:] # training edge index
        data.train_pos_edge_index = torch.stack([r, c], dim=0)

        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge



@deprecated("use 'transforms.RandomLinkSplit' instead")
def do_edge_split_attribute_typeedge_neg_per_edge(
    data: 'torch_geometric.data.Data',
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_type=None,
    ratio = None,
) -> 'torch_geometric.data.Data':

    row, col = data.edge_index
    edge_attr = data.edge_attr
    mask = row < col

    edge_type_list = data.edge_type
    node_type_list = data.node_type
    assert split_type is not None
    mask_type = torch.tensor([i == split_type for i in edge_type_list])

    resulting_mask = mask & mask_type
    row, col = row[resulting_mask], col[resulting_mask]
    if edge_attr is not None:
        edge_attr = edge_attr[resulting_mask]

    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()  # [:, :5*(n_v + n_t)]
    split_edge['train']['edge_attr'] = data.train_pos_edge_attr.t()

    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_attr'] = data.val_pos_edge_attr.t()

    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_attr'] = data.test_pos_edge_attr.t()
    split_edge = get_negative_for_pos_edges(data, split_edge)

    if ratio is not None:
        split_edge = percent_reduction(split_edge, split_edge['train']['edge'], split_edge['train']['edge_eid'], split_edge['train']['eid_edge_neg'], ratio)
    return split_edge


def percent_reduction(split_edge, pos_train_edge, train_pos_edge_eid, train_eid_neg,  ratio):
    pos_train_edge_c = copy.deepcopy(pos_train_edge)
    train_pos_edge_eid_c = copy.deepcopy(train_pos_edge_eid)
    train_eid_neg_c = copy.deepcopy(train_eid_neg)

    row, col = pos_train_edge_c.t()
    mask = row < col
    dir_row, dir_col = row[mask], col[mask]
    dir_pos_edge = torch.stack([dir_row, dir_col], dim=0).t()
    dir_train_pos_edge_eid = train_pos_edge_eid_c[mask]
    dir_train_eid_neg = train_eid_neg_c[mask]

    train_pos_edge_freq = {}
    train_pos_edge_freq_indlist = {}
    for ind, (i, j) in enumerate(zip(dir_pos_edge, dir_train_pos_edge_eid)):
        # print(ind, i, j)
        i0, i1 = i
        i0 = i0.item()
        i1 = i1.item()
        j = j.item()
        if i0 == j:
            train_pos_edge_freq[i0] = train_pos_edge_freq.get(i0, 0) + 1
            if i0 not in train_pos_edge_freq_indlist:
                train_pos_edge_freq_indlist[i0] = list()
            train_pos_edge_freq_indlist[i0].append(ind)
            # train_pos_edge_freq_mask.append(ind)
        else:
            train_pos_edge_freq[i1] = train_pos_edge_freq.get(i1, 0) + 1
            if i1 not in train_pos_edge_freq_indlist:
                train_pos_edge_freq_indlist[i1] = list()
            train_pos_edge_freq_indlist[i1].append(ind)

    train_pos_edge_sampled_inds = []
    for k, v in train_pos_edge_freq_indlist.items():
        sample_ = int(train_pos_edge_freq[k] * ratio)
        sample = max(sample_, 1)
        sampled_inds = random.sample(v, k=sample)
        train_pos_edge_sampled_inds.extend(sampled_inds)

    train_pos_edge_sampled_inds = np.array(train_pos_edge_sampled_inds)

    sampled_dir_pos_edge = dir_pos_edge[train_pos_edge_sampled_inds]
    sampled_dir_train_pos_edge_eid = dir_train_pos_edge_eid[train_pos_edge_sampled_inds]
    sampled_dir_train_eid_neg = dir_train_eid_neg[train_pos_edge_sampled_inds]

    rev_sampled_dir_pos_edge = sampled_dir_pos_edge.clone()
    rev_sampled_dir_pos_edge[:, [0, 1]] = rev_sampled_dir_pos_edge[:, [1, 0]]

    sampled_pos_edge = torch.cat([sampled_dir_pos_edge, rev_sampled_dir_pos_edge], dim=0)
    sampled_train_pos_edge_eid = torch.cat([sampled_dir_train_pos_edge_eid, sampled_dir_train_pos_edge_eid], dim=0)
    sampled_train_eid_neg = torch.cat([sampled_dir_train_eid_neg, sampled_dir_train_eid_neg], dim=0)
    device = split_edge['valid']['edge'].device
    split_edge['train']['ratio_edge'] = sampled_pos_edge.to(device)
    split_edge['train']['ratio_edge_eid'] = sampled_train_pos_edge_eid.to(device)
    split_edge['train']['ratio_eid_edge_neg'] = sampled_train_eid_neg.to(device)
    return split_edge


def z_standard_attribute_from_omics_values(filenames):

    if filenames is None:
        filenames = ['../pkl/attr_matrix_unchanged_ep.pkl', '../pkl/attr_node_list_row_p.pkl',
                     '../pkl/attr_matrix_zscore_em.pkl',  '../pkl/attr_node_list_row_m.pkl']
    # ## em
    em_omics = pickle.load(open('../pkl/patient_em_values.pkl', 'rb'))
    em_patient_list = []
    patient_em = []
    print('normalize em patient', len(list(em_omics.keys())))

    for patient, em_value in em_omics.items():

        em_patient_list.append(patient)
        patient_em.append(em_value)

    patient_em_matrix = np.vstack(patient_em)
    transformed_tensor = np.log(patient_em_matrix + 1)
    masked_array = np.ma.masked_equal(transformed_tensor, 0)

    mean = np.ma.mean(masked_array, axis=0)
    std = np.ma.std(masked_array, axis=0)
    std = np.where(std == 0, 1, std)

    z_type_em = (masked_array - mean) / std
    patient_em_matrix = z_type_em.filled(0)

    # ## ep
    ep_omics = pickle.load(open('../pkl/patient_ep_values.pkl', 'rb'))
    ep_patient_list = []
    patient_ep = []
    print('normalize ep patient', len(list(ep_omics.keys())))
    for patient, ep_value in ep_omics.items():
        # print(np.sum(np.isnan(ep_value)), ep_value.shape)

        ep_patient_list.append(patient)
        patient_ep.append(ep_value)

    patient_ep_matrix = np.vstack(patient_ep)

    pickle.dump(patient_ep_matrix, open(filenames[0], 'wb'))
    pickle.dump(ep_patient_list, open(filenames[1], 'wb'))
    pickle.dump(patient_em_matrix, open(filenames[2], 'wb'))
    pickle.dump(em_patient_list, open(filenames[3], 'wb'))
    return


def z_standard_attribute(data, target_edge_type,  reverse_node_mappings, filenames):
    def edge2valueMatrix(edge_list, num_nodes, edge_value):
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        # todo check if the adjacency is complete or specific to certain group {training set}
        row, col = edge_list
        mask = row < col
        row, col = row[mask], col[mask]
        edges = torch.stack([row, col], dim=0)
        values = edge_value[mask]

        for edge0, edge1, value in zip(edges[0], edges[1], values):
            adjacency_matrix[edge0][edge1] = value
            adjacency_matrix[edge1][edge0] = value
        return adjacency_matrix  # edge_pointer

    def get_node_per_type(reverse_node_mappings, type):
        type_node_list = []

        for node_idx, ori_node in reverse_node_mappings.items():
            if ori_node.startswith(type):
                type_node_list.append(node_idx)
        return type_node_list

    def get_z_score(edges, attr, num_nodes, rowtype, coltype, save_unchanged, filenames):
 
        type_node_list_row = get_node_per_type(reverse_node_mappings, rowtype)
        type_node_list_col = get_node_per_type(reverse_node_mappings, coltype)
        attr_matrix = edge2valueMatrix(edges, num_nodes, attr)  # symmetric matrix of values
        attr_matrix_c = copy.deepcopy(attr_matrix)
        if filenames is None:
            filenames = ['../pkl/attr_matrix_unchanged_ep.pkl', '../pkl/node_list_col_m.pkl',
                     '../pkl/node_list_row_e.pkl', '../pkl/attr_matrix_zscore_em.pkl']

        if save_unchanged:
            type_node_list_col_unchanged = get_node_per_type(reverse_node_mappings, 'p_')
            type_attr_matrix_unchanged = attr_matrix_c[type_node_list_row][:, type_node_list_col_unchanged]
            pickle.dump(type_attr_matrix_unchanged, open(filenames[0], 'wb'))
            pickle.dump(type_node_list_col_unchanged, open(filenames[1], 'wb'))
            pickle.dump(type_node_list_row, open(filenames[2], 'wb'))

        type_attr_matrix = attr_matrix_c[type_node_list_row][:, type_node_list_col]
        assert type_attr_matrix.shape[0] == len(type_node_list_row)
        assert type_attr_matrix.shape[1] == len(type_node_list_col)
        if coltype == 'm_':
            null_exists = True

            if not null_exists:
                transformed_tensor = np.log(type_attr_matrix + 1)
                mean = np.mean(transformed_tensor, axis=0)
                std = np.std(transformed_tensor, axis=0)

                z_type_attr_matrix = (transformed_tensor - mean) / std
                attr_matrix[type_node_list_row][:, type_node_list_col] = z_type_attr_matrix

            else:
                transformed_tensor = np.log(type_attr_matrix + 1)

                masked_array = np.ma.masked_equal(transformed_tensor, 0)

                mean = np.ma.mean(masked_array, axis=0)
                std = np.ma.std(masked_array, axis=0)

                std = np.where(std == 0, 1, std)

                z_type_attr_matrix = (masked_array - mean) / std

                z_type_attr_matrix = z_type_attr_matrix.filled(0)
                attr_matrix[type_node_list_row][:, type_node_list_col] = z_type_attr_matrix

            pickle.dump(z_type_attr_matrix, open(filenames[3], 'wb'))

        z_attrs = []
        for edge0, edge1 in zip(edges[0], edges[1]):
            z_attrs.append(attr_matrix[edge0][edge1])
        z_attrs = torch.tensor(z_attrs)
        return edges, z_attrs, filenames

    if target_edge_type == 1:
        rowtype = 'e_'
        coltype = 'm_'

    num_nodes = data.num_nodes
    edges, z_attrs, filenames = get_z_score(data.edge_index, data.edge_attr, num_nodes, rowtype=rowtype, coltype=coltype, save_unchanged=True, filenames=filenames)

    return edges, z_attrs, filenames


def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

