import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from maskgae.mask import *
import random
from torch.utils.data import DataLoader
import pickle


def generate_negatives_for_evaluations_balance(split_edge, batch_size, valid_neg_num, test_neg_num, device):
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    valid_pos_edge_eid = split_edge['valid']['edge_eid'].to(device)
    valid_pos_edge_eid_neg = split_edge['valid']['eid_edge_neg'].to(device)

    valid_pos_edge_freq = {}
    valid_pos_edge_freq_mask = []
    for ind, (i, j) in enumerate(zip(pos_valid_edge, valid_pos_edge_eid)):
        i0, i1 = i
        i0 = i0.item()
        i1 = i1.item()
        j = j.item()
        # print(i0, i1, j)
        if i0 == j:
            if valid_pos_edge_freq.get(i1, 0) >= 10:
                continue
            valid_pos_edge_freq[i1] = valid_pos_edge_freq.get(i1, 0) + 1
            valid_pos_edge_freq_mask.append(ind)
        else:
            if valid_pos_edge_freq.get(i0, 0) >= 10:
                continue
            valid_pos_edge_freq[i0] = valid_pos_edge_freq.get(i0, 0 ) + 1
            valid_pos_edge_freq_mask.append(ind)

    valid_loader = DataLoader(valid_pos_edge_freq_mask,  batch_size)  #m .to(device)
    valid_loader_list = list(valid_loader)

    pos_valid_edge_src_batch_list = [valid_pos_edge_eid[val_perm] for val_perm in valid_loader_list]
    valid_edge_src_neg_batch_list = [valid_pos_edge_eid_neg[val_perm] for val_perm in valid_loader_list]

    valid_neg_edge_batch_list = []
    for i in range(len(valid_loader_list)):
        pos_valid_edge_src_batch = pos_valid_edge_src_batch_list[i]
        valid_edge_src_neg_batch = valid_edge_src_neg_batch_list[i]
        negs = []
        for e, neg_row in zip(pos_valid_edge_src_batch, valid_edge_src_neg_batch):
            col_index = torch.randint(0, neg_row.size(0), (1, valid_neg_num))
            col_value = neg_row[col_index]
            #  Randomly choose one index.
            es = e.repeat(valid_neg_num).reshape(1, -1, 1)
            es_cols = torch.cat([es, col_value.unsqueeze(-1)], dim=2).to(device)

            negs.append(es_cols)  # for each batch, the negs should be batch, 100
        edge = torch.cat(negs, dim=0).to(device)   # 128, 100, 2
        valid_neg_edge_batch_list.append(edge)

    # test
    pos_test_edge = split_edge['test']['edge'].to(device)
    test_pos_edge_eid = split_edge['test']['edge_eid'].to(device)
    test_pos_edge_eid_neg = split_edge['test']['eid_edge_neg'].to(device)

    test_edge_end_freq = {}
    test_edge_end_freq_mask = []
    for ind, (i, j) in enumerate(zip(pos_test_edge, test_pos_edge_eid)):
        i0, i1 = i
        i0 = i0.item()
        i1 = i1.item()
        j = j.item()
        # print(i0, i1, j)
        if i0 == j:
            if test_edge_end_freq.get(i1, 0) >= 10:
                continue
            test_edge_end_freq[i1] = test_edge_end_freq.get(i1, 0) + 1
            test_edge_end_freq_mask.append(ind)
        else:
            if test_edge_end_freq.get(i0, 0) >= 10:
                continue
            test_edge_end_freq[i0] = test_edge_end_freq.get(i0, 0) + 1
            test_edge_end_freq_mask.append(ind)

    test_loader = DataLoader(test_edge_end_freq_mask, batch_size)  #.to(device)
    test_loader_list = list(test_loader)

    pos_test_edge_src_batch_list = [test_pos_edge_eid[val_perm] for val_perm in test_loader_list]
    test_edge_src_neg_batch_list = [test_pos_edge_eid_neg[val_perm] for val_perm in test_loader_list]

    test_neg_edge_batch_list = []
    for i in range(len(test_loader_list)):
        pos_test_edge_src_batch = pos_test_edge_src_batch_list[i]
        test_edge_src_neg_batch = test_edge_src_neg_batch_list[i]
        negs = []
        for e, neg_row in zip(pos_test_edge_src_batch, test_edge_src_neg_batch):
            col_index = torch.randint(0, neg_row.size(0), (1, test_neg_num))    #  Randomly choose one index.
            col_value = neg_row[col_index]
            es = e.repeat(test_neg_num).reshape(1, -1, 1)
            es_cols = torch.cat([es, col_value.unsqueeze(-1)], dim=2).to(device)

            negs.append(es_cols)  # for each batch, the negs should be batch, 100
        edge = torch.cat(negs, dim=0).to(device)   # 128, 100, 2
        test_neg_edge_batch_list.append(edge)

    return valid_loader_list, test_loader_list, valid_neg_edge_batch_list, test_neg_edge_batch_list


def generate_negatives_for_evaluations_moda(split_edge, batch_size, valid_neg_num, test_neg_num, device):
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    valid_loader = DataLoader(range(pos_valid_edge.size(0)), batch_size)  #m .to(device)
    valid_loader_list = list(valid_loader)
    valid_pos_edge_eid = split_edge['valid']['edge_eid'].to(device)
    valid_pos_edge_eid_neg = split_edge['valid']['eid_edge_neg'].to(device)
    pos_valid_edge_src_batch_list = [valid_pos_edge_eid[val_perm] for val_perm in valid_loader_list]
    valid_edge_src_neg_batch_list = [valid_pos_edge_eid_neg[val_perm] for val_perm in valid_loader_list]

    valid_neg_edge_batch_list = []
    for i in range(len(valid_loader_list)):
        pos_valid_edge_src_batch = pos_valid_edge_src_batch_list[i]
        valid_edge_src_neg_batch = valid_edge_src_neg_batch_list[i]
        negs = []
        for e, neg_row in zip(pos_valid_edge_src_batch, valid_edge_src_neg_batch):
            col_index = torch.randint(0, neg_row.size(0), (1, valid_neg_num))
            col_value = neg_row[col_index]
            #  Randomly choose one index.
            es = e.repeat(valid_neg_num).reshape(1, -1, 1)
            es_cols = torch.cat([es, col_value.unsqueeze(-1)], dim=2).to(device)

            negs.append(es_cols)  # for each batch, the negs should be batch, 100
        edge = torch.cat(negs, dim=0).to(device)   # 128, 100, 2
        valid_neg_edge_batch_list.append(edge)

    # test
    pos_test_edge = split_edge['test']['edge'].to(device)
    test_pos_edge_eid = split_edge['test']['edge_eid'].to(device)
    test_pos_edge_eid_neg = split_edge['test']['eid_edge_neg'].to(device)

    a = open('../pkl/all_codes_em_idx_3_256.pkl', 'rb')

    b = pickle.load(a).to(device)
    mask_with_mod = [i for i, j in enumerate(test_pos_edge_eid) if j in b]

    test_loader = DataLoader(mask_with_mod, batch_size)  #.to(device)
    test_loader_list = list(test_loader)

    pos_test_edge_src_batch_list = [test_pos_edge_eid[val_perm] for val_perm in test_loader_list]
    test_edge_src_neg_batch_list = [test_pos_edge_eid_neg[val_perm] for val_perm in test_loader_list]

    test_neg_edge_batch_list = []
    for i in range(len(test_loader_list)):
        pos_test_edge_src_batch = pos_test_edge_src_batch_list[i]
        test_edge_src_neg_batch = test_edge_src_neg_batch_list[i]
        negs = []
        for e, neg_row in zip(pos_test_edge_src_batch, test_edge_src_neg_batch):
            col_index = torch.randint(0, neg_row.size(0), (1, test_neg_num))  
            col_value = neg_row[col_index]
            es = e.repeat(test_neg_num).reshape(1, -1, 1)
            es_cols = torch.cat([es, col_value.unsqueeze(-1)], dim=2).to(device)

            negs.append(es_cols) 
        edge = torch.cat(negs, dim=0).to(device)   # 128, 100, 2
        test_neg_edge_batch_list.append(edge)
    return valid_loader_list, test_loader_list, valid_neg_edge_batch_list, test_neg_edge_batch_list


def generate_negatives_for_evaluations(split_edge, batch_size, valid_neg_num, test_neg_num, device):
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    valid_loader = DataLoader(range(pos_valid_edge.size(0)), batch_size)  #m .to(device)
    valid_loader_list = list(valid_loader)

    valid_pos_edge_eid = split_edge['valid']['edge_eid'].to(device)
    valid_pos_edge_eid_neg = split_edge['valid']['eid_edge_neg'].to(device)
    pos_valid_edge_src_batch_list = [valid_pos_edge_eid[val_perm] for val_perm in valid_loader_list]
    valid_edge_src_neg_batch_list = [valid_pos_edge_eid_neg[val_perm] for val_perm in valid_loader_list]

    valid_neg_edge_batch_list = []
    for i in range(len(valid_loader_list)):
        pos_valid_edge_src_batch = pos_valid_edge_src_batch_list[i]
        valid_edge_src_neg_batch = valid_edge_src_neg_batch_list[i]
        negs = []
        for e, neg_row in zip(pos_valid_edge_src_batch, valid_edge_src_neg_batch):
            col_index = torch.randint(0, neg_row.size(0), (1, valid_neg_num))
            col_value = neg_row[col_index]
            #  Randomly choose one index.
            es = e.repeat(valid_neg_num).reshape(1, -1, 1)
            es_cols = torch.cat([es, col_value.unsqueeze(-1)], dim=2).to(device)

            negs.append(es_cols)  # for each batch, the negs should be batch, 100
        edge = torch.cat(negs, dim=0).to(device)   # 128, 100, 2
        valid_neg_edge_batch_list.append(edge)

    # test
    pos_test_edge = split_edge['test']['edge'].to(device)
    test_pos_edge_eid = split_edge['test']['edge_eid'].to(device)
    test_pos_edge_eid_neg = split_edge['test']['eid_edge_neg'].to(device)

    test_loader = DataLoader(range(pos_test_edge.size(0)), batch_size)  #.to(device)
    test_loader_list = list(test_loader)
    pos_test_edge_src_batch_list = [test_pos_edge_eid[val_perm] for val_perm in test_loader_list]
    test_edge_src_neg_batch_list = [test_pos_edge_eid_neg[val_perm] for val_perm in test_loader_list]

    test_neg_edge_batch_list = []
    for i in range(len(test_loader_list)):
        pos_test_edge_src_batch = pos_test_edge_src_batch_list[i]
        test_edge_src_neg_batch = test_edge_src_neg_batch_list[i]
        negs = []
        for e, neg_row in zip(pos_test_edge_src_batch, test_edge_src_neg_batch):
            col_index = torch.randint(0, neg_row.size(0), (1, test_neg_num))   #  Randomly choose one index.
            col_value = neg_row[col_index]
            es = e.repeat(test_neg_num).reshape(1, -1, 1)
            es_cols = torch.cat([es, col_value.unsqueeze(-1)], dim=2).to(device)

            negs.append(es_cols)  # for each batch, the negs should be batch, 100
        edge = torch.cat(negs, dim=0).to(device)   # 128, 100, 2
        test_neg_edge_batch_list.append(edge)

    return valid_loader_list, test_loader_list, valid_neg_edge_batch_list, test_neg_edge_batch_list


@torch.no_grad()
def test_transductive_precompute(args, model, predictor, data, split_edge, evaluator, batch_size, encoder_name, dataset,
                                 device, valid_time_data=None, test_time_data=None, valid_batch_list=None, test_batch_list=None,
                                 valid_neg_list=None, test_neg_list=None):
    model.eval()
    predictor.eval()

    if encoder_name == 'mlp':
        if args.minibatch:
            valid_h = model(valid_time_data.x.to("cuda"))
        else:
            valid_h = model(valid_time_data.x)
    else:
        if args.encoder == 'rgcn' or args.encoder == 'prgcn':
            valid_h = model(valid_time_data.x.squeeze(), valid_time_data.adj_t, valid_time_data.adj_edge_type.squeeze())
        else:
            valid_h = model(valid_time_data.x.squeeze(), valid_time_data.adj_t)

    if encoder_name == 'mlp':
        if args.minibatch:
            test_h = model(test_time_data.x.to("cuda"))
        else:
            test_h = model(test_time_data.x)
    else:
        if args.encoder == 'rgcn' or args.encoder == 'prgcn':
            test_h = model(test_time_data.x.squeeze(), test_time_data.adj_t, test_time_data.adj_edge_type.squeeze())
        else:
            test_h = model(test_time_data.x.squeeze(), test_time_data.adj_t)

    pos_valid_edge = split_edge['valid']['edge'].to(test_h.device)
    pos_test_edge = split_edge['test']['edge'].to(test_h.device)

    if args.task == 'em':
        if args.scaled:
            pos_valid_attr = split_edge['valid']['scaled_edge_attr'].squeeze().to(test_h.device)
            pos_test_attr = split_edge['test']['scaled_edge_attr'].squeeze().to(test_h.device)

    test_runs = 150
    # valid evaluation

    pos_valid_preds = []
    # val_perm_list = []
    if args.task == 'em':
        batch_pos_valid_attrs = []
    for i, perm in enumerate(valid_batch_list):
        if i >= test_runs:
            break
        edge = pos_valid_edge[perm].t()
        if args.rank:
            pos_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]
        else:
            pos_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]

        if args.task == 'em':
            edgeattr = pos_valid_attr[perm]
            batch_pos_valid_attrs += [edgeattr.squeeze().cpu()]

    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    if args.task == 'em':
        batches_pos_valid_attr = torch.cat(batch_pos_valid_attrs, dim=0)

    pos_test_preds = []

    if args.task == 'em':
        batch_pos_test_attrs = []

    for i, perm in enumerate(test_batch_list):
        if i >= test_runs:
            break
        edge = pos_test_edge[perm].t()
        if args.rank:
            pos_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]
        else:
            pos_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]

        if args.task == 'em':
            edgeattr = pos_test_attr[perm]
            batch_pos_test_attrs += [edgeattr.squeeze().cpu()]

    if args.task == 'em':
        batches_pos_test_attr = torch.cat(batch_pos_test_attrs, dim=0)

    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    if args.task == 'ed':
        neg_valid_preds = []

    if valid_neg_list is not None:
        edge_batch_list = valid_neg_list[:test_runs]
        edge = torch.cat(edge_batch_list, dim=0)   # 12800, 100, 2
        edge_flat = edge.reshape(-1, 2)  # 12800 * 100, 2

        if edge_flat.shape[0] > edge.shape[0]:
            edge_flat = edge_flat.t()
            if args.rank:
                neg_valid_preds += [predictor(valid_h[edge_flat[0]], valid_h[edge_flat[1]]).reshape(edge.shape[0], edge.shape[1]).squeeze().cpu()]  # 12800 * 200
            else:
                neg_valid_preds += [predictor(valid_h[edge_flat[0]], valid_h[edge_flat[1]]).reshape(edge.shape[0], edge.shape[1]).squeeze().cpu()]  # 12800 * 200
        else:
            if args.rank:
                neg_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]

            else:
                neg_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)  # 12800, 200

    if args.task == 'ed':
        neg_test_preds = []

    if test_neg_list is not None:
        edge_batch_list = test_neg_list[:test_runs]
        edge = torch.cat(edge_batch_list, dim=0)  # 12800, 100, 2
        edge_flat = edge.reshape(-1, 2)  # 12800 * 100, 2

        if edge_flat.shape[0] > edge.shape[0]:
            edge_flat = edge_flat.t()
            if args.rank:
                neg_test_preds += [predictor(test_h[edge_flat[0]], test_h[edge_flat[1]]).reshape(edge.shape[0], edge.shape[1], -1).squeeze().cpu()]  # 12800 * 100, 1
            else:
                neg_test_preds += [predictor(test_h[edge_flat[0]], test_h[edge_flat[1]]).reshape(edge.shape[0], edge.shape[1], -1).squeeze().cpu()]  # 12800 * 100, 1
        else:
            if args.rank:
                neg_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]  # 12800 * 100, 1
            else:
                neg_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    if args.task == 'ed':
        if dataset != "collab":
            for K in [10, 20, 50]:
                evaluator.K = K
                valid_hits = evaluator.eval2D({
                    'y_pred_pos': pos_valid_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_valid_pred,
                })[f'hits@{K}']
                test_hits = evaluator.eval2D({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred,
                })[f'hits@{K}']

                results[f'Hits@{K}'] = (valid_hits, test_hits)

            valid_mrr = evaluator.eval2Dmrr({
                    'y_pred_pos': pos_valid_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_valid_pred,
                })['mrr']
            test_mrr = evaluator.eval2Dmrr({
                    'y_pred_pos': pos_test_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_test_pred,
                })['mrr']

            results['mrr'] = (valid_mrr, test_mrr)

        else:
            for K in [10, 30, 100]:
                evaluator.K = K

                valid_hits = evaluator.eval({
                    'y_pred_pos': pos_valid_pred,
                    'y_pred_neg': neg_valid_pred,
                })[f'hits@{K}']
                test_hits = evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred,
                })[f'hits@{K}']

                results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results, [valid_h, test_h]


@torch.no_grad()
def init_test_precompute(args, model, predictor, data, split_edge, evaluator, batch_size, encoder_name, dataset,
                                 device, valid_time_data=None, test_time_data=None, valid_batch_list=None, test_batch_list=None,
                                 valid_neg_list=None, test_neg_list=None):
    model.eval()
    predictor.eval()
    test_time_data = test_time_data.to(device)
    valid_time_data = valid_time_data.to(device)
    if encoder_name == 'mlp':
        if args.minibatch:
            valid_h = model(valid_time_data.x.to("cuda"))
        else:
            valid_h = model(valid_time_data.x)
    else:
        if args.encoder == 'rgcn' or args.encoder == 'prgcn':
            valid_h = model(valid_time_data.x.squeeze(), valid_time_data.adj_t, valid_time_data.adj_edge_type.squeeze())
        else:
            valid_h = model(valid_time_data.x.squeeze(), valid_time_data.adj_t)

    if encoder_name == 'mlp':
        if args.minibatch:
            test_h = model(test_time_data.x.to("cuda"))
        else:
            test_h = model(test_time_data.x)
    else:
        if args.encoder == 'rgcn' or args.encoder == 'prgcn':
            test_h = model(test_time_data.x.squeeze(), test_time_data.adj_t, test_time_data.adj_edge_type.squeeze())
        else:
            test_h = model(test_time_data.x.squeeze(), test_time_data.adj_t)

    pos_valid_edge = split_edge['valid']['edge'].to(test_h.device)
    pos_test_edge = split_edge['test']['edge'].to(test_h.device)

    if args.task == 'em':
        if args.scaled:
            pos_valid_attr = split_edge['valid']['scaled_edge_attr'].squeeze().to(test_h.device)
            pos_test_attr = split_edge['test']['scaled_edge_attr'].squeeze().to(test_h.device)

    test_runs = 150
    # valid evaluation

    pos_valid_preds = []
    # val_perm_list = []
    if args.task == 'em':
        batch_pos_valid_attrs = []
    for i, perm in enumerate(valid_batch_list): 
        if i >= test_runs:
            break
        edge = pos_valid_edge[perm].t()
        if args.rank:

            pos_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]
        else:
            pos_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]

        if args.task == 'em':
            edgeattr = pos_valid_attr[perm]
            batch_pos_valid_attrs += [edgeattr.squeeze().cpu()]

    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    if args.task == 'em':
        batches_pos_valid_attr = torch.cat(batch_pos_valid_attrs, dim=0)

    pos_test_preds = []

    if args.task == 'em':
        batch_pos_test_attrs = []

    for i, perm in enumerate(test_batch_list):
        if i >= test_runs:
            break
        edge = pos_test_edge[perm].t()
        if args.rank:
            pos_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]
        else:
            pos_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]

        if args.task == 'em':
            edgeattr = pos_test_attr[perm]
            batch_pos_test_attrs += [edgeattr.squeeze().cpu()]

    if args.task == 'em':
        batches_pos_test_attr = torch.cat(batch_pos_test_attrs, dim=0)

    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    # test_neg_runs = 1
    if args.task == 'ed':
        neg_valid_preds = []

    if valid_neg_list is not None:
        edge_batch_list = valid_neg_list[:test_runs]
        edge = torch.cat(edge_batch_list, dim=0)   # 12800, 100, 2
        edge_flat = edge.reshape(-1, 2)  # 12800 * 100, 2

        if edge_flat.shape[0] > edge.shape[0]:
            edge_flat = edge_flat.t()
            if args.rank:
                neg_valid_preds += [predictor(valid_h[edge_flat[0]], valid_h[edge_flat[1]]).reshape(edge.shape[0], edge.shape[1]).squeeze().cpu()]  # 12800 * 200

            else:
                neg_valid_preds += [predictor(valid_h[edge_flat[0]], valid_h[edge_flat[1]]).reshape(edge.shape[0], edge.shape[1]).squeeze().cpu()]  # 12800 * 200
        else:
            if args.rank:
                neg_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]

            else:
                neg_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)  # 12800, 200

    if args.task == 'ed':
        neg_test_preds = []
    if test_neg_list is not None:

        edge_batch_list = test_neg_list[:test_runs]
        edge = torch.cat(edge_batch_list, dim=0)   # 12800, 100, 2
        edge_flat = edge.reshape(-1, 2)  # 12800 * 100, 2

        if edge_flat.shape[0] > edge.shape[0]:
            edge_flat = edge_flat.t()

            if args.rank:
                neg_test_preds += [predictor(test_h[edge_flat[0]], test_h[edge_flat[1]]).reshape(edge.shape[0], edge.shape[1], -1).squeeze().cpu()]  # 12800 * 100, 1

            else:
                neg_test_preds += [predictor(test_h[edge_flat[0]], test_h[edge_flat[1]]).reshape(edge.shape[0], edge.shape[1], -1).squeeze().cpu()]  # 12800 * 100, 1
        else:
            if args.rank:
                neg_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]  # 12800 * 100, 1
            else:
                neg_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

    if args.task == 'ed':
        valid_hits_list = []
        test_hits_list = []
        valid_mrr_list = []
        test_mrr_list = []
        if dataset != "collab":
            for K in [10, 20, 50]:
                evaluator.K = K
                valid_hits = evaluator.eval2D({
                    'y_pred_pos': pos_valid_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_valid_pred,
                })[f'hits@{K}']
                test_hits = evaluator.eval2D({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred,
                })[f'hits@{K}']
                valid_hits_list.append(valid_hits)
                test_hits_list.append(test_hits)

            valid_mrr = evaluator.eval2Dmrr({
                    'y_pred_pos': pos_valid_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_valid_pred,
                })['mrr']
            test_mrr = evaluator.eval2Dmrr({
                    'y_pred_pos': pos_test_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_test_pred,
                })['mrr']

            valid_mrr_list.append(valid_mrr)
            test_mrr_list.append(test_mrr)

        else:
            for K in [10, 30, 100]:
                evaluator.K = K

                valid_hits = evaluator.eval({
                    'y_pred_pos': pos_valid_pred,
                    'y_pred_neg': neg_valid_pred,
                })[f'hits@{K}']
                test_hits = evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred,
                })[f'hits@{K}']




@torch.no_grad()
def test_transductive(args, model, predictor, data, split_edge, evaluator, batch_size, encoder_name, dataset, device, valid_time_data=None, test_time_data=None):
    model.eval()
    predictor.eval()

    if encoder_name == 'mlp':
        if args.minibatch:
            valid_h = model(valid_time_data.x.to("cuda"))
        else:
            valid_h = model(valid_time_data.x)
    else:
        if args.encoder == 'rgcn' or args.encoder == 'prgcn':
            valid_h = model(valid_time_data.x.squeeze(), valid_time_data.adj_t, valid_time_data.adj_edge_type.squeeze())
        else:
            valid_h = model(valid_time_data.x.squeeze(), valid_time_data.adj_t)

    if encoder_name == 'mlp':
        if args.minibatch:
            test_h = model(test_time_data.x.to("cuda"))
        else:
            test_h = model(test_time_data.x)
    else:
        if args.encoder == 'rgcn' or args.encoder == 'prgcn':
            test_h = model(test_time_data.x.squeeze(), test_time_data.adj_t, test_time_data.adj_edge_type.squeeze())
        else:
            test_h = model(test_time_data.x.squeeze(), test_time_data.adj_t)

    pos_valid_edge = split_edge['valid']['edge'].to(test_h.device)
    pos_test_edge = split_edge['test']['edge'].to(test_h.device)

    valid_pos_edge_eid = split_edge['valid']['edge_eid']
    valid_pos_edge_eid_neg = split_edge['valid']['eid_edge_neg']

    test_pos_edge_eid = split_edge['test']['edge_eid']
    test_pos_edge_eid_neg = split_edge['test']['eid_edge_neg']

    if args.task == 'em':
        if args.scaled:
            pos_valid_attr = split_edge['valid']['scaled_edge_attr'].squeeze().to(test_h.device)
            pos_test_attr = split_edge['test']['scaled_edge_attr'].squeeze().to(test_h.device)

    test_runs = 100
    # valid evaluation

    pos_valid_preds = []
    val_perm_list = []
    if args.task == 'em':
        batch_pos_valid_attrs = []
    for i, perm in enumerate(DataLoader(range(pos_valid_edge.size(0)), batch_size)):
        if i >= test_runs:
            break
        edge = pos_valid_edge[perm].t()
        if args.rank:
            pos_scores_valid = torch.mul(valid_h[edge[0]], valid_h[edge[1]])
            pos_scores_valid = torch.sum(pos_scores_valid, dim=1).squeeze().cpu()
            pos_valid_preds += [pos_scores_valid]
        else:
            pos_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]

        if args.task == 'em':
            edgeattr = pos_valid_attr[perm]
            batch_pos_valid_attrs += [edgeattr.squeeze().cpu()]
        val_perm_list.append(perm)

    # sample negs
    val_perm_list = torch.cat(val_perm_list, dim=0)
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    if args.task == 'em':
        batches_pos_valid_attr = torch.cat(batch_pos_valid_attrs, dim=0)

    pos_test_preds = []
    test_perm_list = []

    if args.task == 'em':
        batch_pos_test_attrs = []

    for i, perm in enumerate(DataLoader(range(pos_test_edge.size(0)), batch_size)):
        if i >= test_runs:
            break
        edge = pos_test_edge[perm].t()
        if args.rank:
            pos_scores_test = torch.mul(test_h[edge[0]], test_h[edge[1]])
            pos_scores_test = torch.sum(pos_scores_test, dim=1).squeeze().cpu()  # the larger the better
            pos_test_preds += [pos_scores_test]
        else:
            pos_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]

        if args.task == 'em':
            edgeattr = pos_test_attr[perm]
            batch_pos_test_attrs += [edgeattr.squeeze().cpu()]

        test_perm_list.append(perm)

    if args.task == 'em':
        batches_pos_test_attr = torch.cat(batch_pos_test_attrs, dim=0)

    # input as perm_list
    test_perm_list = torch.cat(test_perm_list, dim=0)
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    # test_neg_runs = 1
    if args.task == 'ed':
        neg_valid_preds = []

        pos_valid_edge_src_batch = valid_pos_edge_eid[val_perm_list]
        valid_edge_src_neg_batch = valid_pos_edge_eid_neg[val_perm_list]

        negs = []  # n 200
        for e, neg_row in zip(pos_valid_edge_src_batch, valid_edge_src_neg_batch):
            col_index = torch.randint(0, neg_row.size(0), (1, 200))   # Randomly choose one index.
            es = e.repeat(200).reshape(1, -1, 1)
            es_cols = torch.cat([es, col_index.unsqueeze(-1)], dim=2)

            negs.append(es_cols)

        edge = torch.cat(negs, dim=0)   # 12800, 100, 2

        edge_flat = edge.reshape(-1, 2)  # 12800 * 100, 2

        if edge_flat.shape[0] > edge.shape[0]:
            edge_flat = edge_flat.t()
            if args.rank:
                neg_scores_valid = torch.mul(valid_h[edge_flat[0]], valid_h[edge_flat[1]])  # 12800 * 100, 2
                neg_scores_valid = torch.sum(neg_scores_valid, dim=1).squeeze().cpu()  #
                neg_scores_valid = neg_scores_valid.reshape(edge.shape[0], 200, -1).squeeze().cpu()  # 12800, 100, 1
                neg_valid_preds += [neg_scores_valid]
            else:
                neg_valid_preds += [predictor(valid_h[edge_flat[0]], valid_h[edge_flat[1]]).reshape(edge.shape[0], 200).squeeze().cpu()]  # 12800 * 200
            # neg_valid_pred = torch.cat(neg_valid_preds, dim=0)  # (12800, 100, )
        else:
            if args.rank:
                neg_scores_valid = torch.mul(valid_h[edge[0]], valid_h[edge[1]])
                neg_scores_valid = torch.sum(neg_scores_valid, dim=1).squeeze().cpu()
                neg_valid_preds += [neg_scores_valid]
            else:
                neg_valid_preds += [predictor(valid_h[edge[0]], valid_h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)  # 12800, 200

        #  test
        pos_test_edge_src_batch = test_pos_edge_eid[test_perm_list]
        test_edge_src_neg_batch = test_pos_edge_eid_neg[test_perm_list]

        negs = []  # n 200
        neg_test_preds = []
        for e, neg_row in zip(pos_test_edge_src_batch, test_edge_src_neg_batch):
            col_index = torch.randint(0, neg_row.size(0), (1, 300))   # Randomly choose one index.
            es = e.repeat(300).reshape(1, -1, 1)
            es_cols = torch.cat([es, col_index.unsqueeze(-1)], dim=2)
            negs.append(es_cols)  #

        edge = torch.cat(negs, dim=0)   # 12800, 100, 2
        edge_flat = edge.reshape(-1, 2)

        if edge_flat.shape[0] > edge.shape[0]:
            edge_flat = edge_flat.t()

            if args.rank:
                neg_scores_test = torch.mul(test_h[edge_flat[0]], test_h[edge_flat[1]])  # 12800 * 200, 2
                neg_scores_test = torch.sum(neg_scores_test, dim=1).squeeze().cpu()  #
                neg_scores_test = neg_scores_test.reshape(edge.shape[0], 300, -1).squeeze().cpu()  # 128, 100, 1
                neg_test_preds += [neg_scores_test]
            else:
                neg_test_preds += [predictor(test_h[edge_flat[0]], test_h[edge_flat[1]]).reshape(edge.shape[0], 300, -1).squeeze().cpu()]  # 12800 * 100, 1
        else:
            if args.rank:
                neg_scores_test = torch.mul(test_h[edge[0]], test_h[edge[1]])
                neg_scores_test = torch.sum(neg_scores_test, dim=1).squeeze().cpu()
                neg_test_preds += [neg_scores_test]
            else:
                neg_test_preds += [predictor(test_h[edge[0]], test_h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    if args.task == 'ed':
        if dataset != "collab":
            for K in [10, 20, 50]:
                evaluator.K = K
                valid_hits = evaluator.eval2D({
                    'y_pred_pos': pos_valid_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_valid_pred,
                })[f'hits@{K}']
                test_hits = evaluator.eval2D({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred,
                })[f'hits@{K}']
                results[f'Hits@{K}'] = (valid_hits, test_hits)

            valid_mrr = evaluator.eval2Dmrr({
                    'y_pred_pos': pos_valid_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_valid_pred,
                })['mrr']
            test_mrr = evaluator.eval2Dmrr({
                    'y_pred_pos': pos_test_pred,  # (12800,) v.s. (12800, 200)
                    'y_pred_neg': neg_test_pred,
                })['mrr']

            results['mrr'] = (valid_mrr, test_mrr)

        else:
            for K in [10, 30, 100]:
                evaluator.K = K

                valid_hits = evaluator.eval({
                    'y_pred_pos': pos_valid_pred,
                    'y_pred_neg': neg_valid_pred,
                })[f'hits@{K}']
                test_hits = evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred,
                })[f'hits@{K}']

                results[f'Hits@{K}'] = (valid_hits, test_hits)
    return results, [valid_h, test_h]
