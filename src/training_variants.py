import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from extra_loss import *


def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    if mat.size(0) != mat.size(1):
        edge_index = mat
        num_edges = edge_index.size(1)
        num_edges_to_keep = int((1 - dropout) * num_edges)
        perm = torch.randperm(num_edges)
        edges_to_keep = perm[:num_edges_to_keep]
        edge_index_droped = edge_index[:, edges_to_keep]
        return edge_index_droped
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)


def feature_distillation_loss(features_student, features_teacher, alpha=0.5):
    mse_loss = nn.MSELoss()
    distillation_loss = mse_loss(features_student, features_teacher)
    return distillation_loss

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


def train(args, model, predictor, data, split_edge, optimizer, model_l, predictor_l, data_l, optimizer_l, batch_size,
                  encoder_name, dataset, transductive, device, new_adj_t, new_adj_t_type, pos_edges_l,  neg_edges_l,
                  new_adj_tl, new_adj_t_typel, dropadj=False, p_patient=None, m_patient=None):
    
    fc1 = nn.Sequential(
        nn.Linear(64, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 64, bias=True),
        )
    fc1.to(device)

    if transductive == "transductive":
        data = data.to(device)
        if args.ratio is None:
            pos_train_edge = split_edge['train']['edge'].to(device)
        else:
            pos_train_edge = split_edge['train']['ratio_edge'].to(device)

    if args.task == 'ed':
        if args.ratio is None:
            train_pos_edge_eid = split_edge['train']['edge_eid'].to(device)
            train_pos_edge_eid_neg = split_edge['train']['eid_edge_neg'].to(device)
        else:
            train_pos_edge_eid = split_edge['train']['ratio_edge_eid'].to(device)
            train_pos_edge_eid_neg = split_edge['train']['ratio_eid_edge_neg'].to(device)

    model.train()
    predictor.train()
    if args.task == 'ed':
        margin_rank_loss = nn.MarginRankingLoss(margin=args.margin).to(device)
        bce_loss = nn.BCELoss().to(device)
    if args.task == 'em':
        lossF = mse_loss = nn.MSELoss().to(device)

    total_loss = total_examples = 0
    train_iters = 600

    model_l.train()
    predictor_l.train()
    total_lossl = total_examplesl = 0
    data_l = data_l.to(device)
    pos_train_edge_l = pos_edges_l.to(device)
    neg_train_edge_l = neg_edges_l.to(device)

    pc_batch_size = int(pos_train_edge_l.size()[1] / 1)

    for zz in range(0,5):
        for il, perml in enumerate(DataLoader(range(pos_train_edge_l.size(1)), pc_batch_size, shuffle=True)):

            if il > train_iters:
                break
            optimizer_l.zero_grad()

            if encoder_name == 'mlp':
                hl = model_l(data_l.x)
            else:
                if transductive == "transductive":
                    if dropadj is True:
                        hl = model_l(data_l.x.squeeze(), sparse_dropout(new_adj_tl, 5 * args.dropout))
                    else:
                        hl = model_l(data_l.x.squeeze(), new_adj_tl)

            batch_edge_l = pos_train_edge_l[:, perml]
            batch_neg_edge_l = neg_train_edge_l[:, perml]

            pos_pc_outl = predictor_l(hl[batch_edge_l[0]], hl[batch_edge_l[1]]).squeeze()
            neg_pc_outl = predictor_l(hl[batch_neg_edge_l[0]], hl[batch_neg_edge_l[1]]).squeeze()
            pc_targl = torch.ones(batch_edge_l.size(1)).to(device)
            pc_lossl = margin_rank_loss(pos_pc_outl, neg_pc_outl, pc_targl)

            pc_lossl.backward()
            torch.nn.utils.clip_grad_norm_(model_l.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor_l.parameters(), 1.0)

            optimizer_l.step()
            num_examplesl = batch_edge_l.size(1)
            total_lossl += pc_lossl.item() * num_examplesl
            total_examplesl += num_examplesl
            avg_lossl = total_lossl / total_examplesl

    p_kd_patient = data_l.newpatient[p_patient]
    m_kd_patient = data_l.newpatient[m_patient] #  index of the new patient
    kd_patient = torch.cat((p_kd_patient, m_kd_patient), dim=0)

    avg_kdloss = 0

    for i, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)):
        if i > train_iters:
            break
        optimizer.zero_grad()

        if encoder_name == 'mlp':
            h = model(data.x)
        else:
            if transductive == "transductive":
                if args.encoder == 'rgcn' or args.encoder == 'prgcn':
                    h = model(data.x.squeeze(), new_adj_t, new_adj_t_type.squeeze())
                else:
                    if dropadj is True:
                        h = model(data.x.squeeze(), sparse_dropout(new_adj_t, 3 * args.dropout))
                    else:
                        h = model(data.x.squeeze(), new_adj_t)

        edge = pos_train_edge[perm].t()

        if args.task == 'ed':
            train_edge_eid_batch = train_pos_edge_eid[perm].to(device)
            train_edge_eid_neg_batch = train_pos_edge_eid_neg[perm].to(device)
            negs = []
            for e, neg_row in zip(train_edge_eid_batch, train_edge_eid_neg_batch):
                col_index = torch.randint(0, neg_row.size(0), (1,))  # Randomly choose one index.
                negs.append((e, neg_row[col_index]))  #
            neg_edge = torch.tensor(negs).t().to(device)  # 2 batch

            if args.rank:
                pos_out = predictor(h[edge[0]], h[edge[1]]).squeeze()
                neg_out = predictor(h[neg_edge[0]], h[neg_edge[1]]).squeeze()
                targ = torch.ones(edge.size(1)).to(device)
                loss = margin_rank_loss(pos_out, neg_out, targ)

            else:
                train_edges = torch.cat((edge, neg_edge), dim=-1)
                train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
                out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
                out = F.sigmoid(out)
                loss = bce_loss(out, train_label)

        for fi, permf in enumerate(DataLoader(kd_patient, pc_batch_size, shuffle=True)):
            batch_pold = data_l.oldpatient[permf]  #
            batch_pnew = data_l.newpatient[permf]  #

            if encoder_name == 'mlp':
                hf = model(data.x)
            else:
                if transductive == "transductive":

                    if dropadj is True:
                        hf = model(data.x.squeeze(), sparse_dropout(new_adj_t, 3 * args.dropout))
                    else:
                        hf = model(data.x.squeeze(), new_adj_t)

            with torch.no_grad():
                if encoder_name == 'mlp':
                    hlf = model_l(data_l.x)
                else:
                    if transductive == "transductive":

                        if dropadj is True:
                            hlf = model_l(data_l.x.squeeze(), sparse_dropout(new_adj_tl, 3 * args.dropout))
                        else:
                            hlf = model_l(data_l.x.squeeze(), new_adj_tl)

            f_pold = hf[batch_pold]  
            f_pnew = hlf[batch_pnew]  
            kd_loss_mse = F.mse_loss(f_pold, f_pnew)

            A_embedding = f_pold
            B_embedding = f_pnew

            tau = 0.8    # default = 0.8
            f = lambda x: torch.exp(x / tau)
            A_embedding = fc1(A_embedding).squeeze()
            B_embedding = fc1(B_embedding).squeeze()
            refl_sim = f(sim(A_embedding, A_embedding))
            between_sim = f(sim(A_embedding, B_embedding))

            loss_1 = -torch.log(
                between_sim.diag()
                / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
            ret = loss_1
            kd_loss = ret.mean()

        l2loss = torch.norm(model.node_embedding.weight, p=2, dim=-1).mean()

        t_loss = loss + args.kd*kd_loss + args.kd_mse*kd_loss_mse + 0.00001*l2loss

        t_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()
        num_examples = edge.size(1)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        avg_loss = total_loss / total_examples

    return avg_loss, avg_lossl, avg_kdloss
