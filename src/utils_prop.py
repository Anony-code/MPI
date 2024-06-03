import sys
sys.path.append('.')
sys.path.append('..')
import torch
import copy
import math


def get_2codebook_kd_nodes(data, card_book, node_num, patient_codes, patient_list, edge_types):
    edge_indexes = []
    edge_types_list = []
    # edge_type: e-pc: 4
    # edge_type: e-mc: 5

    emask = (data.node_type == 0).squeeze()

    old_patient = torch.arange(0, data.x.shape[0])[emask].tolist()
    new_patient = torch.arange(0, len(old_patient)).tolist()

    dict_patient = dict(zip(old_patient, new_patient))

    has_patientId = [False] * len(new_patient)
    if len(edge_types) > 1:
        assert len(patient_list) > 1 and len(patient_codes) > 1
        for i in range(len(edge_types)):
            for e, ecode in zip(patient_list[i], patient_codes[i]):
                pnode_idx = dict_patient[e]
                has_patientId[pnode_idx] = True
                for book_, code_ in enumerate(ecode):
                    code = code_ + node_num + book_ * card_book

                    edge_indexes.append((pnode_idx, code))
                    edge_types_list.append(edge_types[i])
                    edge_indexes.append((code, pnode_idx))
                    edge_types_list.append(edge_types[i])
    else:
        for e, ecode in zip(patient_list[0], patient_codes[0]):
            pnode_idx = dict_patient[e]
            has_patientId[pnode_idx] = True

            for book_, code_ in enumerate(ecode):
                code = code_ + node_num + book_ * card_book

                edge_indexes.append((pnode_idx, code))
                edge_types_list.append(edge_types[0])
                edge_indexes.append((code, pnode_idx))
                edge_types_list.append(edge_types[0])
    edge_indexes = torch.tensor(edge_indexes).t()
    edge_types_list = torch.tensor(edge_types_list).reshape(-1, 1)
    return edge_indexes, edge_types_list, has_patientId


def get_2code_kd_samemsg(args, data, split_edge, split_type, code_edge1, code_edge2, code_edge_type1, code_edge_type2, card_book_p, card_book, n_book, ratio=None, keep_type=[], keep_type_code=[]):

    '''for right part'''
    data_r = copy.deepcopy(data)
    split_edge_c = copy.deepcopy(split_edge)
    if ratio is None:
        signal_edge = split_edge_c['train']['edge'].t()
    else:
        signal_edge = split_edge_c['train']['ratio_edge'].t()

    signal_edge_type = torch.tensor([split_type] * signal_edge.shape[-1]).reshape(-1, 1)

    all_prop = signal_edge
    all_prop_type = signal_edge_type
    data_r.adj_t = all_prop
    data_r.adj_edge_type = all_prop_type

    '''for left part'''
    data_l = copy.deepcopy(data)
    pcode_edge_c = copy.deepcopy(code_edge1)
    pcode_edge_type_c = copy.deepcopy(code_edge_type1)
    mcode_edge_c = copy.deepcopy(code_edge2)
    mcode_edge_type_c = copy.deepcopy(code_edge_type2)

    if args.msg_edge_other == 'e-me':
        keep_type = [3]

    all_prop = torch.cat([pcode_edge_c, mcode_edge_c], dim=-1)
    all_prop_type = torch.cat([pcode_edge_type_c, mcode_edge_type_c], dim=0)
    emask = (data_l.node_type == 0).squeeze()
    patient_list = torch.arange(0, data_l.x.shape[0])[emask]
    data_l.newpatient = torch.arange(0, patient_list.shape[0]).reshape(-1, 1)
    data_l.node_type = torch.tensor([0] * data_l.newpatient.shape[0]).reshape(-1, 1)

    data_l.oldpatient = patient_list.reshape(-1, 1)
    node_add = torch.tensor(range(data_l.newpatient.shape[0], data_l.newpatient.shape[0] + (card_book_p + card_book) * n_book)).reshape(-1, 1)
    data_l.x = torch.cat([data_l.newpatient, node_add], dim=0)
    pnode_add_type = torch.tensor([5] * (card_book_p * n_book)).reshape(-1, 1)
    mnode_add_type = torch.tensor([6] * (card_book * n_book)).reshape(-1, 1)
    data_l.node_type = torch.cat([data_l.node_type, pnode_add_type, mnode_add_type], dim=0)
    data_l.edge_index = all_prop
    data_l.edge_attr = None
    data_l.edge_type = all_prop_type
    data_l.adj_t = all_prop
    data_l.adj_edge_type = all_prop_type

    return data_r, data_l
