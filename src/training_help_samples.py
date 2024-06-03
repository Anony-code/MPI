import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from maskgae.mask import *
import random


def get_source_nodes(data, srcNode = 0):
        node_type_list = data.node_type
        node_type_list = node_type_list.tolist()

        num_nodes = len(node_type_list)
        get_src_node = [i[0] == srcNode for i in node_type_list]
        node_list = np.arange(num_nodes)
        src_nodes = node_list[get_src_node]
        return src_nodes  # this is a numpy


def generate_pc_for_epoch(args, data, split_edge, n_codebook, card_codebook, message_all, alter_edge_index=None):
    def myMask(args, undirected):
        if args.mask_type == 'path':
            mask = MaskPath(p=args.p, num_nodes=data.num_nodes,
                            start=args.start,
                            walk_length=args.encoder_layers+1)
        elif args.mask_type == 'edge':
            mask = MaskEdge(p=args.p, undirected=undirected)
        else:
            mask = None  # vanilla GAE
        return mask

    # only mask one time for one epoch
    def mask_once_for_epoch(data, alt_edges,  messageall = True):
        if args.mask is True:
            # get pc edges
            all_type = data.adj_edge_type
            if data.adj_t.shape[-1] > 2:
                message_edges = alt_edges
            else:
                message_edges = data.adj_t

            keep_pc = [4, 5]
            other_pc_mask = torch.tensor([i != 4 and i != 5 for i in all_type])
            other_pc_edges = message_edges[:, other_pc_mask]
            other_pc_edges_type = all_type[other_pc_mask]

            pc_mask = torch.tensor([i in keep_pc for i in all_type])
            pc_edges = message_edges[:, pc_mask]
            pc_edges_type = all_type[pc_mask]

            row, col = pc_edges
            dir_mask = row < col
            pc_dir_edges = pc_edges[:, dir_mask]
            pc_dir_edges_type = pc_edges_type[dir_mask]
            pc_dir_remaining_edges, pc_dir_masked_edges, pc_dir_bernoli = mask(pc_dir_edges)
            print(pc_dir_masked_edges.shape, pc_dir_remaining_edges.shape, message_edges.shape, all_type.shape)

            if messageall:
                new_adj_t = message_edges
                new_adj_type = all_type
            else:
                rev_pc_dir_remaining_edges = pc_dir_remaining_edges.clone()
                rev_pc_dir_remaining_edges[:, [-2, -1]] = pc_dir_remaining_edges[:, [-1, -2]]
                pc_remaining_edges = torch.cat([pc_dir_remaining_edges, rev_pc_dir_remaining_edges], dim=0)

                pc_dir_remaining_edges_type = pc_dir_edges_type[~pc_dir_bernoli]
                pc_remaining_edges_type = torch.cat([pc_dir_remaining_edges_type, pc_dir_remaining_edges_type], dim=0)

                new_adj_t = torch.cat([other_pc_edges, pc_remaining_edges], dim=-1).to(data.x.device)
                new_adj_type = torch.cat([other_pc_edges_type, pc_remaining_edges_type]).to(data.x.device)

        return new_adj_t, new_adj_type, pc_dir_masked_edges, pc_dir_bernoli, pc_edges

    def sample_code_neg_edges_for_epoch(all_masked_edges, all_pc_edges, patient_eids, code_eids):
        pc_neg_edges = []
        all_masked_edges = all_masked_edges.cpu().numpy()
        mrow, mcol = all_masked_edges
        code_eids = code_eids.cpu().numpy()

        all_pc_edges = all_pc_edges.cpu().numpy().reshape(-1, 2)
        all_pc_edges = list(set([tuple(i) for i in all_pc_edges]))

        order_pc_masked_edges = []
        for ei, ej in zip(mrow, mcol):
            if ei in patient_eids:
                ej_neg = random.choice(code_eids)[0]
                while (ei, ej_neg) in all_pc_edges:
                    ej_neg = random.choice(code_eids)[0]
                pc_neg_edges.append((ei, ej_neg))
                order_pc_masked_edges.append((ei, ej))
            else:
                ei_neg = random.choice(code_eids)[0]
                while (ej, ei_neg) in all_pc_edges:
                    ei_neg = random.choice(code_eids)[0]
                pc_neg_edges.append((ej, ei_neg))
                order_pc_masked_edges.append((ej, ei))

        masked_neg_edges = torch.tensor(pc_neg_edges).t()
        ordered_masked_edges = torch.tensor(order_pc_masked_edges).t()

        return masked_neg_edges, ordered_masked_edges

    mask = myMask(args, undirected=False) # mask will be leaking , so manual single edge filtering

    new_adj_t, new_adj_type, dir_masked_edges, dir_berl, all_pc_edges = mask_once_for_epoch(data, alter_edge_index, message_all)  # mask at both directions, thus there might be leaky
    patient_eids = get_source_nodes(data)
    code_eids = data.x[-n_codebook*card_codebook:]

    neg_edges, ordered_masked_edges = sample_code_neg_edges_for_epoch(dir_masked_edges, all_pc_edges, patient_eids, code_eids)
    perm = torch.randperm(neg_edges.size()[-1])
    perm_order_masked_edges = ordered_masked_edges[:, perm]

    perm_neg_edges = neg_edges[:, perm]
    return new_adj_t, new_adj_type, perm_order_masked_edges, perm_neg_edges


def generate_2pc_for_epoch(args, data, split_edge, n_codebook, p_card_codebook, card_codebook, message_all, alter_edge_index=None):

    def myMask(args, undirected):
        if args.mask_type == 'path':
            mask = MaskPath(p=args.p, num_nodes=data.num_nodes,
                            start=args.start,
                            walk_length=args.encoder_layers+1)
        elif args.mask_type == 'edge':
            mask = MaskEdge(p=args.p, undirected=undirected) 
        else:
            mask = None  # vanilla GAE
        return mask

    def mask_once_for_epoch(data, alt_edges,  messageall = True):
        if args.mask is True:
            all_type = data.adj_edge_type
            if data.adj_t.shape[1] == data.adj_t.shape[0]:
                message_edges = alt_edges
            else:
                message_edges = data.adj_t

            keep_pc = [4, 5]
            other_pc_mask = torch.tensor([i not in keep_pc for i in all_type])
            other_pc_edges = message_edges[:, other_pc_mask]
            other_pc_edges_type = all_type[other_pc_mask]

            pc_mask = torch.tensor([i in keep_pc for i in all_type])
            pc_edges = message_edges[:, pc_mask]
            pc_edges_type = all_type[pc_mask]

            row, col = pc_edges
            dir_mask = row < col
            pc_dir_edges = pc_edges[:, dir_mask]
            pc_dir_edges_type = pc_edges_type[dir_mask]
            pc_dir_remaining_edges, pc_dir_masked_edges, pc_dir_bernoli = mask(pc_dir_edges)

            if messageall: 
                new_adj_t = message_edges
                new_adj_type = all_type
            else:
                rev_pc_dir_remaining_edges = pc_dir_remaining_edges.clone()
                rev_pc_dir_remaining_edges[:, [-2, -1]] = pc_dir_remaining_edges[:, [-1, -2]]
                pc_remaining_edges = torch.cat([pc_dir_remaining_edges, rev_pc_dir_remaining_edges], dim=0)

                pc_dir_remaining_edges_type = pc_dir_edges_type[~pc_dir_bernoli]
                pc_remaining_edges_type = torch.cat([pc_dir_remaining_edges_type, pc_dir_remaining_edges_type], dim=0)

                new_adj_t = torch.cat([other_pc_edges, pc_remaining_edges], dim=-1).to(data.x.device)
                new_adj_type = torch.cat([other_pc_edges_type, pc_remaining_edges_type]).to(data.x.device)

        return new_adj_t, new_adj_type, pc_dir_masked_edges, pc_dir_bernoli, pc_edges

    def sample_code_neg_edges_for_epoch(all_masked_edges, all_pc_edges, patient_eids, code_eids1, code_eids2):
        pc_neg_edges = []
        all_masked_edges = all_masked_edges.cpu().numpy()
        mrow, mcol = all_masked_edges
        # patient_eids = patient_eids
        code_eids1 = code_eids1.cpu().numpy()
        code_eids2 = code_eids2.cpu().numpy()

        all_pc_edges = all_pc_edges.cpu().numpy().reshape(-1, 2)
        all_pc_edges = list(set([tuple(i) for i in all_pc_edges]))

        order_pc_masked_edges = []

        for ei, ej in zip(mrow, mcol):  
            # print(ei, ej)
            if ei in patient_eids:
                # print('ei patient', ei)
                if ej in code_eids1:
                    ej_neg = random.choice(code_eids1)[0]
                    while (ei, ej_neg) in all_pc_edges:
                        ej_neg = random.choice(code_eids1)[0]
                    # print('ej neg', ej_neg)
                    pc_neg_edges.append((ei, ej_neg))
                    order_pc_masked_edges.append((ei, ej))
                else:
                    ej_neg = random.choice(code_eids2)[0]
                    while (ei, ej_neg) in all_pc_edges:
                        ej_neg = random.choice(code_eids2)[0]
                    pc_neg_edges.append((ei, ej_neg))
                    # print('ej neg', ej_neg)

                    order_pc_masked_edges.append((ei, ej))
            else:  # ej in patient_eid:
                if ei in code_eids1:
                    ei_neg = random.choice(code_eids1)[0]
                    while (ej, ei_neg) in all_pc_edges:
                        ei_neg = random.choice(code_eids1)[0]
                    print('ei neg', ei_neg)
                    pc_neg_edges.append((ej, ei_neg))
                    order_pc_masked_edges.append((ej, ei))
                else:
                    print('ei code2', ei)

                    ei_neg = random.choice(code_eids2)[0]
                    while (ej, ei_neg) in all_pc_edges:
                        ei_neg = random.choice(code_eids2)[0]
                    print('ei neg', ei_neg)
                    pc_neg_edges.append((ej, ei_neg))
                    order_pc_masked_edges.append((ej, ei))

        masked_neg_edges = torch.tensor(pc_neg_edges).t()
        ordered_masked_edges = torch.tensor(order_pc_masked_edges).t()

        return masked_neg_edges, ordered_masked_edges

    mask = myMask(args, undirected=False)  # mask will be leaking , so manual single edge filtering
    new_adj_t, new_adj_type, dir_masked_edges, dir_berl, all_pc_edges = mask_once_for_epoch(data, alter_edge_index, message_all)  # mask at both directions, thus there might be leaky
    patient_eids = get_source_nodes(data, 0)
    code_eids1 = data.x[ -n_codebook * p_card_codebook - n_codebook * card_codebook:-1 * n_codebook*card_codebook]
    code_eids2 = data.x[-1 * n_codebook*card_codebook:]

    neg_edges, ordered_masked_edges = sample_code_neg_edges_for_epoch(dir_masked_edges, all_pc_edges, patient_eids, code_eids1, code_eids2)

    # pc_batch_size = masked_edges.size(0) / batch_number
    perm = torch.randperm(neg_edges.size()[-1])
    perm_order_masked_edges = ordered_masked_edges[:, perm]

    # perm_masked_edges = dir_masked_edges[:, perm]
    perm_neg_edges = neg_edges[:, perm]
    return new_adj_t, new_adj_type, perm_order_masked_edges, perm_neg_edges


def simple_generate_2pc_for_epoch(args, data, split_edge, n_codebook, p_card_codebook, card_codebook, message_all, alter_edge_index=None, pc_typemask=None):

    def mask_once_for_epoch(data, alt_edges,  messageall = True):
        if args.mask is True:
            if data.adj_t.shape[1] == data.adj_t.shape[0]:  
                message_edges = alt_edges
            else:
                message_edges = data.adj_t
            pc_edges = message_edges[:, pc_typemask]

            row, col = pc_edges
            dir_mask = row < col

            pc_dir_edges = pc_edges[:, dir_mask]

            maskperm = torch.randperm(pc_dir_edges.size()[-1])[: int(args.p * pc_dir_edges.size()[-1])]
            pc_dir_masked_edges = pc_dir_edges[:, maskperm]

            if messageall:
                new_adj_t = message_edges
                new_adj_type = data.adj_edge_type

        return new_adj_t, new_adj_type, pc_dir_masked_edges, None, pc_edges

    # change adj_t for once, and sample negative for once, and then add one more prediction loss
    def sample_code_neg_edges_for_epoch(all_masked_edges, all_pc_edges, patient_eids, code_eids1, code_eids2):
        pc_neg_edges = []
        all_masked_edges = all_masked_edges.cpu().numpy()
        mrow, mcol = all_masked_edges

        code_eids1 = code_eids1.cpu().numpy()
        code_eids2 = code_eids2.cpu().numpy()

        all_pc_edges = all_pc_edges.cpu().numpy().reshape(-1, 2)
        all_pc_edges = list(set([tuple(i) for i in all_pc_edges]))

        order_pc_masked_edges = []

        for ei, ej in zip(mrow, mcol):  
            if ei in patient_eids:
                print('ei patient', ei)
                if ej in code_eids1:
                    ej_neg = random.choice(code_eids1)[0]
                    while (ei, ej_neg) in all_pc_edges:
                        ej_neg = random.choice(code_eids1)[0]
                    # print('ej neg', ej_neg)
                    pc_neg_edges.append((ei, ej_neg))
                    order_pc_masked_edges.append((ei, ej))
                else:
                    ej_neg = random.choice(code_eids2)[0]
                    while (ei, ej_neg) in all_pc_edges:
                        ej_neg = random.choice(code_eids2)[0]
                    pc_neg_edges.append((ei, ej_neg))
                    # print('ej neg', ej_neg)

                    order_pc_masked_edges.append((ei, ej))
            else:  # ej in patient_eid:
                if ei in code_eids1:
                    ei_neg = random.choice(code_eids1)[0]
                    while (ej, ei_neg) in all_pc_edges:
                        ei_neg = random.choice(code_eids1)[0]
                    print('ei neg', ei_neg)
                    pc_neg_edges.append((ej, ei_neg))
                    order_pc_masked_edges.append((ej, ei))
                else:
                    print('ei code2', ei)

                    ei_neg = random.choice(code_eids2)[0]
                    while (ej, ei_neg) in all_pc_edges:
                        ei_neg = random.choice(code_eids2)[0]
                    print('ei neg', ei_neg)
                    pc_neg_edges.append((ej, ei_neg))
                    order_pc_masked_edges.append((ej, ei))

        masked_neg_edges = torch.tensor(pc_neg_edges).t()
        ordered_masked_edges = torch.tensor(order_pc_masked_edges).t()

        return masked_neg_edges, ordered_masked_edges

    new_adj_t, new_adj_type, dir_masked_edges, dir_berl, all_pc_edges = mask_once_for_epoch(data, alter_edge_index, message_all)

    patient_eids = get_source_nodes(data, 0)
    code_eids1 = data.x[-n_codebook*(p_card_codebook + card_codebook):-1 * n_codebook*card_codebook]
    code_eids2 = data.x[-1 * n_codebook*card_codebook:]

    neg_edges, ordered_masked_edges = sample_code_neg_edges_for_epoch(dir_masked_edges, all_pc_edges, patient_eids, code_eids1, code_eids2)

    perm = torch.randperm(neg_edges.size()[-1])
    perm_order_masked_edges = ordered_masked_edges[:, perm]
    perm_neg_edges = neg_edges[:, perm]
    return new_adj_t, new_adj_type, perm_order_masked_edges, perm_neg_edges


def prepare_generate_for_kd(args, data, split_edge, n_codebook, pcard_codebook, card_codebook, message_all, alter_edge_index=None, pc_typemask=None):
    def mask_once_for_epoch(data, alt_edges, messageall=True):
        if data.adj_t.shape[1] == data.adj_t.shape[0]: 
            message_edges = alt_edges
        else:
            message_edges = data.adj_t

        pc_edges = message_edges[:, pc_typemask]

        row, col = pc_edges
        dir_mask = row < col

        pc_dir_edges = pc_edges[:, dir_mask]
        # maskperm = torch.randperm(pc_dir_edges.size()[-1])[: int(args.p * pc_dir_edges.size()[-1])]
        maskperm = torch.randperm(pc_dir_edges.size()[-1])
        pc_dir_masked_edges = pc_dir_edges[:, maskperm]
        if messageall:
            new_adj_t = message_edges
            new_adj_type = data.adj_edge_type
        return new_adj_t, new_adj_type, pc_dir_masked_edges

    def sample_code_neg_edges_for_epoch(all_masked_edges, all_pc_edges, patient_eids, code_eids1, code_eids2):
        pc_neg_edges = []
        all_masked_edges = all_masked_edges.cpu().numpy()
        mrow, mcol = all_masked_edges

        code_eids1 = code_eids1.cpu().numpy()
        code_eids2 = code_eids2.cpu().numpy()

        all_pc_edges = all_pc_edges.cpu().numpy().reshape(-1, 2)
        all_pc_edges = list(set([tuple(i) for i in all_pc_edges]))

        order_pc_masked_edges = []

        for ei, ej in zip(mrow, mcol):  
            if ei in patient_eids:
                if ej in code_eids1:
                    ej_neg = random.choice(code_eids1)[0]
                    while (ei, ej_neg) in all_pc_edges:
                        ej_neg = random.choice(code_eids1)[0]
                    pc_neg_edges.append((ei, ej_neg))
                    order_pc_masked_edges.append((ei, ej))
                else:
                    ej_neg = random.choice(code_eids2)[0]
                    while (ei, ej_neg) in all_pc_edges:
                        ej_neg = random.choice(code_eids2)[0]
                    pc_neg_edges.append((ei, ej_neg))
                    order_pc_masked_edges.append((ei, ej))
            else:
                if ei in code_eids1:
                    ei_neg = random.choice(code_eids1)[0]
                    while (ej, ei_neg) in all_pc_edges:
                        ei_neg = random.choice(code_eids1)[0]
                    pc_neg_edges.append((ej, ei_neg))
                    order_pc_masked_edges.append((ej, ei))
                else:
                    ei_neg = random.choice(code_eids2)[0]
                    while (ej, ei_neg) in all_pc_edges:
                        ei_neg = random.choice(code_eids2)[0]
                    pc_neg_edges.append((ej, ei_neg))
                    order_pc_masked_edges.append((ej, ei))

        masked_neg_edges = torch.tensor(pc_neg_edges).t()
        ordered_masked_edges = torch.tensor(order_pc_masked_edges).t()

        return masked_neg_edges, ordered_masked_edges

    new_adj_t, new_adj_type, dir_masked_edges = mask_once_for_epoch(data, alter_edge_index, message_all)

    patient_eids = get_source_nodes(data, 0)
    code_eids1 = data.x[-(pcard_codebook + card_codebook) * n_codebook:-1 * n_codebook * card_codebook]
    code_eids2 = data.x[-1 * n_codebook * card_codebook:]

    neg_edges, ordered_masked_edges = sample_code_neg_edges_for_epoch(dir_masked_edges, dir_masked_edges, patient_eids, code_eids1, code_eids2)

    perm = torch.randperm(neg_edges.size()[-1])
    perm_order_masked_edges = ordered_masked_edges[:, perm]
    perm_neg_edges = neg_edges[:, perm]
    return new_adj_t, new_adj_type, perm_order_masked_edges, perm_neg_edges
