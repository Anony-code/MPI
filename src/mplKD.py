import os
import sys
import torch.autograd
sys.path.append('.')
sys.path.append('..')

from predlink_dataset import LinkPredictionDataset
from parser import parse_args
from torch_geometric.nn import SAGEConv
from os.path import exists
from local_evaluate import Evaluator
from logger import Logger, ProductionLogger
import datetime

from models import GCN, SAGE, GIN, LinkPredictor, Adj2GNNinit, Adj2GNN
from utils import *
from utils_prop import *
from maskgae.mask import *
# from extra_loss import *

from training_variants import train  
from training_help_samples import prepare_generate_for_kd, get_source_nodes
from testing_variants import generate_negatives_for_evaluations, test_transductive_precompute, init_test_precompute


def process_adj_sugrl(data):
    from sugrl.data_unit.utils import normalize_graph
    i = torch.LongTensor([data.adj_t[0].numpy(), data.adj_t[1].numpy()])
    v = torch.FloatTensor(torch.ones([data.adj_t.shape[-1]]))
    A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
    A = A_sp.to_dense()
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    A_I_nomal = normalize_graph(A_I)
    A_I_nomal = A_I_nomal.to_sparse()  # has self-loop
    return A_I_nomal


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_at_valid(args, data, split_edge, device):
    data_c = copy.deepcopy(data)
    split_edge_c = copy.deepcopy(split_edge)
    add_adj = torch.cat([data_c.adj_t.t(), split_edge_c['train']['edge']], dim=0).t()
    add_adj_type = torch.cat([data_c.adj_edge_type.reshape(-1, 1), torch.tensor( [args.split_type] * split_edge_c['train']['edge'].shape[-1]).reshape(-1, 1)], dim=0)
    data_c.adj = add_adj
    data_c.adj_type = add_adj_type
    return data_c.to(device)


def add_at_test(args, data, split_edge, device):
    data_c = copy.deepcopy(data)
    split_edge_c = copy.deepcopy(split_edge)
    add_adj = torch.cat([data_c.adj_t.t(), split_edge_c['train']['edge'], split_edge_c['valid']['edge']], dim=0).t()
    add_adj_type = torch.cat([data_c.adj_edge_type.reshape(-1, 1), torch.tensor( [args.split_type] * (split_edge_c['train']['edge'].shape[-1] + split_edge_c['train']['edge'].shape[-1])).reshape(-1, 1)], dim=0)
    data_c.adj = add_adj
    data_c.adj_type = add_adj_type
    return data_c.to(device)

def return_new_neg(spe_neg_, new_fre):
    temp_list = []
    temp = []
    for item in spe_neg_:
        for ie in item:
            if ie in new_fre.keys():
                temp.append(ie)
        temp_list.append(len(temp))
        temp = []
    t_m = min(temp_list)

    li = []
    li_l = []
    for item in spe_neg_:
        for ie in item:
            if ie in new_fre.keys() and len(li_l) < t_m:
                li_l.append(ie)
        li.append(li_l)
        li_l = []
    return li


def data_formulation(data, split_edge, args):
    dd = data.edge_index.numpy()
    dic_fre = {}
    for i in range(0, len(dd[0])):
        if dd[0][i] not in dic_fre:
            dic_fre[dd[0][i]] = [dd[1][i]]
        else:
            dic_fre[dd[0][i]].append(dd[1][i])
    
    fre = {}
    for k in dic_fre.keys():
        fre[k] = len(dic_fre[k])
    new_fre = {}
    for k in fre.keys():
        if fre[k] < args.fre_filter:
            new_fre[k] = fre[k]
    
    totall = 0
    for k in new_fre.keys():
        totall += new_fre[k]
    
    arr_1 = []
    arr_2 = []

    for k in dic_fre.keys():
        if k in new_fre.keys():
            for v in dic_fre[k]:
                if v in new_fre.keys():
                    arr_1.append(k)
                    arr_2.append(v)

    arr = [arr_1, arr_2]
    arr = torch.as_tensor(arr)

    data.edge_index = arr
    data.edge_attr = data.edge_attr[0:arr.size()[1],]
    data.edge_type = data.edge_type[0:arr.size()[1],]
    
    spe = split_edge['train']['ratio_edge'].numpy()
    spe_eid = split_edge['train']['ratio_edge_eid'].numpy()
    spe_neg = split_edge['train']['ratio_eid_edge_neg'].numpy()
    
    spe1 = []
    spe_eid_ = []
    spe_neg_ = []
    mask = []
    for i in range(0, len(spe)):
        if spe[i][0] in new_fre.keys() and spe[i][1] in new_fre.keys():
            mask.append(True)
        else:
            mask.append(False)
            
    spe1 = spe[mask]
    spe_eid_ = spe_eid[mask]
    spe_neg_ = spe_neg[mask]


    li = return_new_neg(spe_neg_, new_fre)

    spe1 = torch.as_tensor(spe1)
    spe_eid_n = torch.as_tensor(spe_eid_)
    spe_neg_n = torch.as_tensor(li)

    split_edge['train']['ratio_edge'] = spe1
    split_edge['train']['ratio_edge_eid'] = spe_eid_n
    split_edge['train']['ratio_eid_edge_neg'] = spe_neg_n

    sev = split_edge['valid']['edge'].numpy()
    sev_eid = split_edge['valid']['edge_eid'].numpy()
    sev_neg = split_edge['valid']['eid_edge_neg'].numpy()

    sev1 = []
    sev_eid_ = []
    sev_neg_ = []
    mask_v = []
    for i in range(0, len(sev)):
        if sev[i][0] in new_fre.keys() and sev[i][1] in new_fre.keys():
            mask_v.append(True)
        else:
            mask_v.append(False)
    sev1 = sev[mask_v]
    sev_eid_ = sev_eid[mask_v]
    sev_neg_ = sev_neg[mask_v]

    sev1 = torch.as_tensor(sev1)
    sev_eid_n = torch.as_tensor(sev_eid_)
    sev_neg_n = torch.as_tensor(return_new_neg(sev_neg_,new_fre))
    split_edge['valid']['edge'] = sev1
    split_edge['valid']['edge_eid'] = sev_eid_n
    split_edge['valid']['eid_edge_neg'] = sev_neg_n


    set = split_edge['test']['edge'].numpy()
    set_eid = split_edge['test']['edge_eid'].numpy()
    set_neg = split_edge['test']['eid_edge_neg'].numpy()

    set1 = []
    set_eid_ = []
    set_neg_ = []
    mask_t = []
    for i in range(0, len(set)):
        if set[i][0] in new_fre.keys() and set[i][1] in new_fre.keys():
            mask_t.append(True)
        else:
            mask_t.append(False)
    set1 = set[mask_t]
    set_eid_ = set_eid[mask_t]
    set_neg_ = set_neg[mask_t]
        
    set1 = torch.as_tensor(set1)
    set_eid_n = torch.as_tensor(set_eid_)
    set_neg_n = torch.as_tensor(return_new_neg(set_neg_, new_fre))


    split_edge['test']['edge'] = set1
    split_edge['test']['edge_eid'] = set_eid_n
    split_edge['test']['eid_edge_neg'] = set_neg_n

    return data, split_edge


def main():
    args = parse_args()
    assert args.ratio is not None

    set_all_seeds(args.seed)

    now = datetime.datetime.now()
    datetime_string = now.strftime("%m%d_%H%M%S")  # e.g., '2024-02-17 14:55:02'
    os.makedirs("./results", exist_ok=True)

    Logger_file = "./results/" + "peredge_{}_{}_lr{}_patience{}_{}_test{}_seed{}_phe".format\
        (args.encoder, args.transductive, str(args.lr)[2:], str(args.patience), args.msg_edge,
         200, str(args.seed)) \
                + '_ratio' + str(args.ratio if args.ratio is not None else 'none').replace('.', '') \
                + '_' + datetime_string + '_' + args.note+".txt"
    print('\nResults saved in ', Logger_file)
    file = open(Logger_file, "a")
    file.write('\n'.join([str(name) + ':' + str(value) for name, value in vars(args).items()]))
    file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.transductive == "transductive":
        if args.dataset == 'ehr':
            nxgraph = pickle.load(open('../pkl/sub20_graph_phe.pkl', 'rb'))   
            graph_dataset = LinkPredictionDataset(nxgraph)
            data = graph_dataset.data

            split_type = 2   
            
            if args.ratio is not None:
                pklfile = "../pkl/" + args.dataset + "_split_seed{}_peredge_trainratio{}_phe.pkl"\
                    .format(str(args.seed), str(args.ratio).replace('.', ''))  
            else:
                pklfile = "../pkl/" + args.dataset + "_split_seed{}_peredge_phe.pkl".format(str(args.seed) )

            if exists(pklfile):
                split_edge = torch.load(pklfile)
            else:
                split_edge = do_edge_split_attribute_typeedge_neg_per_edge(data, 0.1, 0.1, split_type, args.ratio) 
                torch.save(split_edge, pklfile)

            if args.msg_edge == 'ed_kd':  
                if not exists("../pkl/" + "attr_node_list_row_p_phe.pkl"):
                    '''
                    the first stage to generate code by rqvae, if not existing yet. 
                    the numerical values of multi-modal omics will be used to generate rqvae code, conducted in the "rqvae" folder. 
                    '''
                    z_edge_type = 1   
                    if z_edge_type == 1:
                        filenames = ['../pkl/attr_matrix_unchanged_ep_phe.pkl', '../pkl/attr_node_list_row_p_phe.pkl',  '../pkl/attr_matrix_zscore_em_phe.pkl', '../pkl/attr_node_list_row_m_phe.pkl']
                        z_standard_attribute_from_omics_values(filenames)  # the files are from get_omics_values
                        sys.exit()

            input_size = args.num_features
        
            if args.msg_edge == 'ed_kd':
                print('%%%%%%%%%% Second of the Two stages:: 2 use the generated nodes to add into graph')

                card_book = args.meta_card_book
                n_book = args.n_book
                card_book_p = args.prot_card_book

                pcode = pickle.load(open('../pkl/sep_codes_ep_{}_{}_phe.pkl'.format(str(n_book), str(card_book_p)), 'rb'))
                mcode = pickle.load(open('../pkl/sep_codes_em_{}_{}_phe.pkl'.format(str(n_book), str(card_book)), 'rb'))

                idx_filenames = ['../pkl/attr_node_list_row_p_phe.pkl', '../pkl/attr_node_list_row_m_phe.pkl']
                pcode_eidx = pickle.load(open(idx_filenames[0], 'rb'))
                mcode_eidx = pickle.load(open(idx_filenames[1], 'rb'))

                # find patient index in the graph
                pcode_eidx_ = ['e_' + idx for idx in pcode_eidx]
                mcode_eidx_ = ['e_' + idx for idx in mcode_eidx]

                pingraph_mask = [idx in graph_dataset.node_mappings for idx in pcode_eidx_]
                mingraph_mask = [idx in graph_dataset.node_mappings for idx in mcode_eidx_]

                pcode_eidx_ = [pcode_eidx_[idx] for idx, valid in enumerate(pingraph_mask) if valid]
                mcode_eidx_ = [mcode_eidx_[idx] for idx, valid in enumerate(mingraph_mask) if valid]

                pcode_eidx_ingraph = [graph_dataset.node_mappings[idx] for idx in pcode_eidx_]
                mcode_eidx_ingraph = [graph_dataset.node_mappings[idx] for idx in mcode_eidx_]

                pcode_ingraph = pcode[pingraph_mask]
                mcode_ingraph = mcode[mingraph_mask]

                print('%%%%%%%%%% Num of patients with codes: ', 'protein', len(pcode_eidx_ingraph),
                      'metab', len(mcode_eidx_ingraph), 'protein code', pcode_ingraph.shape,
                      'metab code', mcode_ingraph.shape)  # 1518 proteomics patient | 7731 metabolomics patient

                node_num = (data.node_type == 0).sum().item()

                data, split_edge = data_formulation(data, split_edge, args)

                pcode_edge_indexes, pcode_edge_types, has_p_patient = get_2codebook_kd_nodes(data, card_book_p, node_num, [pcode_ingraph], [pcode_eidx_ingraph], [4])
                mcode_edge_indexes, mcode_edge_types, has_m_patient = get_2codebook_kd_nodes(data, card_book, node_num + card_book_p * n_book, [mcode_ingraph], [mcode_eidx_ingraph], [5])

                # build 'data_l' as patient-code graph, alongwith the 'data' as the patient-phenotype graph
                data, data_l = get_2code_kd_samemsg(args, data, split_edge, split_type, pcode_edge_indexes, mcode_edge_indexes, pcode_edge_types,
                                    mcode_edge_types, card_book_p, card_book, n_book, args.ratio, keep_type_code=[4, 5])

            valid_loader_list, test_loader_list, valid_neg_edge_batch_list, test_neg_edge_batch_list =\
            generate_negatives_for_evaluations(split_edge, args.batch_size, 200, 200, device)

            def change_order(args, split_edge):
                if args.ratio is not None:
                    train_pos = split_edge['train']['ratio_edge']
                    train_id = split_edge['train']['ratio_edge_eid']
                    train_neg = split_edge['train']['ratio_eid_edge_neg']
                else:
                    train_pos = split_edge['train']['edge']
                    train_id = split_edge['train']['edge_eid']
                    train_neg = split_edge['train']['eid_edge_neg']

                train_pos_row, train_pos_col = train_pos.t()
                train_pos_mask = train_pos_row < train_pos_col

                train_pos = train_pos[train_pos_mask]
                train_id = train_id[train_pos_mask]
                train_neg = train_neg[train_pos_mask]

                valid_pos = split_edge['valid']['edge']
                valid_id = split_edge['valid']['edge_eid']

                test_pos = split_edge['test']['edge']
                test_id = split_edge['test']['edge_eid']

                order_train_pos = []
                for te, tid in zip(train_pos, train_id):
                    te1, te2 = te
                    te1 = te1.item()
                    te2 = te2.item()
                    tid = tid.item()
                    if te1 == tid:
                        order_train_pos.append((te1, te2))
                    else:
                        order_train_pos.append((te2, te1))
                order_train_pos = np.array(order_train_pos)
                order_train_pos = torch.tensor(order_train_pos).to(train_pos.device)

                order_valid_pos = []
                for te, tid in zip(valid_pos, valid_id):
                    te1, te2 = te
                    te1 = te1.item()
                    te2 = te2.item()
                    tid = tid.item()
                    if te1 == tid:
                        order_valid_pos.append((te1, te2))
                    else:
                        order_valid_pos.append((te2, te1))
                order_valid_pos = np.array(order_valid_pos)
                order_valid_pos = torch.tensor(order_valid_pos).to(train_pos.device)

                order_test_pos = []
                for te, tid in zip(test_pos, test_id):
                    te1, te2 = te
                    if te1 == tid:
                        order_test_pos.append((te1, te2))
                    else:
                        order_test_pos.append((te2, te1))
                order_test_pos = np.array(order_test_pos)
                order_test_pos = torch.tensor(order_test_pos).to(train_pos.device)

                if args.ratio is not None:
                    split_edge['train']['ratio_edge'] = order_train_pos
                    split_edge['train']['ratio_edge_eid'] = train_id
                    split_edge['train']['ratio_eid_edge_neg'] = train_neg

                else:
                    split_edge['train']['edge'] = order_train_pos
                    split_edge['train']['edge_id'] = train_id
                    split_edge['train']['eid_edge_neg'] = train_neg

                split_edge['valid']['edge'] = order_valid_pos
                split_edge['test']['edge'] = order_test_pos
                return split_edge
            split_edge = change_order(args, split_edge)
            
            if args.use_valedges_as_input: # not applied
                val_edge_index = split_edge['valid']['edge'].t()
                full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
                data.full_adj_t = full_edge_index
            else:
                data.full_adj_t = data.adj_t
            

    if args.encoder == 'sage':
        num_nodes = data.x.shape[0]
        model = SAGE(args.dataset, num_nodes, input_size, args.cfg[0],
                    args.cfg[1], args.num_layers,
                    args.dropout, SAGEConv).to(device)
        predictor = LinkPredictor(args.predictor, 2 * args.cfg[1], args.cfg[1], 1, 2, args.dropout).to(device)

    elif args.encoder == 'gin':
        num_nodes = data.x.shape[0]
        model = GIN(args.dataset, num_nodes, input_size, args.cfg[0],
                    args.cfg[1],
                    args.dropout ).to(device)
        predictor = LinkPredictor(args.predictor, 2 * args.cfg[1], args.cfg[1], 1, 2, args.dropout).to(device)

    elif args.encoder == 'adj2gnn':
        A_I_nomal = process_adj_sugrl(data).to(device)
        data.alt_adj_t = data.adj_t
        data.adj_t = A_I_nomal
        if args.code_init and args.msg_edge != 'ed_kd' :
            code_embs1 = pickle.load(open('../pkl/sep_codes_embs_ep_{}_{}_phe.pkl'.format(str(n_book), str(args.prot_card_book)), 'rb'))
            code_embs2 = pickle.load(open('../pkl/sep_codes_embs_em_{}_{}_phe.pkl'.format(str(n_book), str(args.meta_card_book)), 'rb'))
            code_embs1 = code_embs1.to(device)
            code_embs2 = code_embs2.to(device)
            model = Adj2GNNinit(num_nodes=data.x.shape[0] - n_book *(args.prot_card_book  + args.meta_card_book), in_channels=input_size, in_channels2=32,
                                            init_fea1=code_embs1, init_fea2=code_embs2, cfg=args.cfg, dropout=args.dropout).to(device)
        else:
            model = Adj2GNN(num_nodes=data.x.shape[0], in_channels=input_size, cfg=args.cfg,
                        dropout=args.dropout).to(device)
            
        predictor = LinkPredictor(args.predictor, 2 * args.cfg[-1],  args.cfg[-1], 1, 2, args.dropout).to(device)
        print('\tdata adjacency norm processing: ', data.adj_t.shape)

  
    if args.msg_edge == 'ed_kd':
        if args.encoder == 'adj2gnn':
            A_I_nomal_l = process_adj_sugrl(data_l).to(device)
            data_l.alt_adj_t = data_l.adj_t
            data_l.adj_t = A_I_nomal_l

            if args.code_init:
                code_embs1 = pickle.load(open('../pkl/sep_codes_embs_ep_{}_{}_phe.pkl'.format(str(n_book), str(args.prot_card_book)), 'rb'))
                code_embs2 = pickle.load(open('../pkl/sep_codes_embs_em_{}_{}_phe.pkl'.format(str(n_book), str(args.meta_card_book)), 'rb'))
                code_embs1 = code_embs1.to(device)
                code_embs2 = code_embs2.to(device)

                model_l = Adj2GNNinit(num_nodes=data_l.x.shape[0] - n_book *(args.prot_card_book  + args.meta_card_book), in_channels=input_size, in_channels2=32,
                                                        init_fea1=code_embs1, init_fea2=code_embs2, cfg=args.cfg, dropout=args.dropout).to(device)
            else:
                model_l = Adj2GNN(num_nodes=data_l.x.shape[0], in_channels=input_size, cfg=args.cfg,
                            dropout=args.dropout).to(device)

        predictor_l = LinkPredictor(args.predictor, 2 * args.cfg[-1],  args.cfg[-1], 1, 2, args.dropout).to(device)

        print('\tdata_l adjacency norm processing: ', data_l.adj_t.shape)

    evaluator = Evaluator(meta_info_path='master.csv', name='ehr')
    if args.transductive == "transductive":

        if args.dataset == 'ehr' and args.task != 'em':
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@20': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'mrr':Logger(args.runs, args)
            }
        elif args.dataset == 'ehr' and args.task == 'em':
            loggers = {
                'mae': Logger(args.runs, args),
                'rmse': Logger(args.runs, args),
            }
        else:
            loggers = {
                'Hits@10': ProductionLogger(args.runs, args),
                'Hits@20': ProductionLogger(args.runs, args),
                'Hits@30': ProductionLogger(args.runs, args),
                'Hits@50': ProductionLogger(args.runs, args),
                'AUC': ProductionLogger(args.runs, args),
            }

    if args.use_valedges_as_input:
        valid_time_data = add_at_valid(args, data, split_edge, device)
        test_time_data = add_at_test(args, data, split_edge, device)
    else:
        valid_time_data = data
        test_time_data = data

    for run in range(args.runs):
        torch_geometric.seed.seed_everything(args.runseed + run)  
        model.reset_parameters()
        predictor.reset_parameters()

        if args.msg_edge == 'ed_kd':
            model_l.reset_parameters()
            predictor_l.reset_parameters()

        optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(predictor.parameters()), lr=args.lr, weight_decay=args.weight_decay)
                                                                                                                                                                                             
        optimizer_l = torch.optim.Adam(
        list(model_l.parameters()) +
        list(predictor_l.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        all_type_l = data_l.adj_edge_type
        keep_pc = [4, 5]
        left_typemask = torch.tensor([i in keep_pc for i in all_type_l]) 

        cnt_wait = 0
        best_val = 0.0
        best_val1 = best_val2 = best_val3 = 0.0

        init_test_precompute(args, model, predictor, data, split_edge,
                    evaluator, args.batch_size, args.encoder, args.dataset, device, valid_time_data,
                                                  test_time_data, valid_loader_list, test_loader_list,
                                                  valid_neg_edge_batch_list, test_neg_edge_batch_list)

        for epoch in range(1, 1 + args.epochs):
            if args.transductive == "transductive":
                new_adj_t = data.adj_t
                new_adj_t_type = data.adj_edge_type
                new_adj_tl = data_l.adj_t
                new_adj_t_typel = data_l.adj_edge_type  # 其实这里没有改变adj_t

                _, _, l_generate_pc_pos, l_generate_pc_neg = prepare_generate_for_kd(args, data_l, None, args.n_book,
                                                                                args.meta_card_book, args.prot_card_book, message_all=True,
                                                                                alter_edge_index=data_l.alt_adj_t, pc_typemask=left_typemask)

                loss, loss_l, loss_kd = train(args, model, predictor, data, split_edge, optimizer, model_l,
                                                        predictor_l, data_l, optimizer_l, args.batch_size, args.encoder,
                                                        args.dataset, args.transductive, device, new_adj_t, new_adj_t_type,
                                                        l_generate_pc_pos,  l_generate_pc_neg, new_adj_tl, new_adj_t_typel,
                                                        dropadj=args.dropadj, p_patient=has_p_patient, m_patient=has_m_patient)

                results, h = test_transductive_precompute(args, model, predictor, data, split_edge,
                            evaluator, args.batch_size, args.encoder, args.dataset, device, valid_time_data,
                                                          test_time_data, valid_loader_list, test_loader_list,
                                                          valid_neg_edge_batch_list, test_neg_edge_batch_list)

            if args.task == 'em':
                pass
            else:
                if args.allbest:  # not applied
                    if results[args.metric][0] >= best_val1 or results[args.metric][1] >= best_val2 or results[args.metric][2] >= best_val3:
                        best_val1 = max(results[args.metric][0], best_val1)
                        best_val2 = max(results[args.metric][1], best_val2)
                        best_val3 = max(results[args.metric][2], best_val3)
                        cnt_wait = 0
                    else:
                        cnt_wait += 1
                else:
                    if results[args.metric][0] >= best_val:  
                        best_val = results[args.metric][0]
                        cnt_wait = 0
                    else:
                        cnt_wait += 1

            for key, result in results.items():
                loggers[key].add_result(run, result) 

            if epoch == 1 or epoch % args.log_steps == 0:
                print('---')
                if args.task != 'em':
                    if args.transductive == "transductive":
                        for key, result in results.items():
                            valid_hits, test_hits = result
                            print('Key', key)
                            print(f'Run: {run :02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Loss: {loss:.4f}, '
                                    f'Valid: {100 * valid_hits:.2f}%, '
                                    f'Test: {100 * test_hits:.2f}%')
                    else:
                        for key, result in results.items():
                            valid_hits, test_hits, old_old, old_new, new_new = result
                            print('Key', key)
                            print(f'Run: {run :02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'valid: {100 * valid_hits:.2f}%, '
                                f'test: {100 * test_hits:.2f}%, '
                                f'old_old: {100 * old_old:.2f}%, '
                                f'old_new: {100 * old_new:.2f}%, '
                                f'new_new: {100 * new_new:.2f}%')
                else:
                    if args.transductive == 'transductive':
                        for key, result in results.items():
                            val_result, test_result = result
                            print('-------- Logging Epoch: Key {} --------'.format(key))
                            print(f'\tRun: {run :02d}, '
                                        f'Epoch: {epoch:02d}, '
                                        f'Train Loss: {loss:.4f}, '
                                        f'Valid : {val_result:.3f}, '
                                        f'Test: {test_result:.3f}')

            if cnt_wait >= args.patience: #and epoch > 300:
                print('######## Break for run {}: epoch {}; cnt_wait {} ########'.format(run, epoch, cnt_wait))
                break

        for key in loggers.keys():
            if args.task == 'ed':
                loggers[key].print_statistics(run)  # for this run, on all the epoch
            elif args.task == 'em':
                loggers[key].print_statistics_regression(run)  # for this run, on all the epoch

    file = open(Logger_file, "a")  # loggers[key] record every run and every epoch result
    file.write(f'\n-------- Loggers (over Runs):\n')
    for key in loggers.keys():
        if args.task == 'em':
            if args.transductive == "transductive":
                file.write(f'{key}:\n')
                best_results = []
                for r in loggers[key].results: # r is the run numbers
                    r = torch.tensor(r)
                    valid = r[:, 0].min().item()
                    test1 = r[r[:, 0].argmin(), 1].item()
                    best_results.append((valid, test1))  # best of all runs

                best_result = torch.tensor(best_results)
                r = best_result[:, 1]  # best for all runs
                file.write(f'Test: {r.mean():.4f} ± {r.std():.4f}\n')

        else:
            if args.transductive == "transductive":
                file.write(f'{key}:\n')
                best_results = []
                for r in loggers[key].results:
                    r = 100 * torch.tensor(r)
                    valid = r[:, 0].max().item()
                    test1 = r[r[:, 0].argmax(), 1].item()
                    best_results.append((valid, test1))
                    print(key, 'key best at', r[:, 0].argmax())
                best_result = torch.tensor(best_results)

                r = best_result[:, 1]
                file.write(f'Test: {r.mean():.4f} ± {r.std():.4f}\n')
            else:
                file.write(f'{key}:\n')
                best_results = []
                for r in loggers[key].results:
                    r = 100 * torch.tensor(r)
                    val = r[r[:, 0].argmax(), 0].item()
                    test_r = r[r[:, 0].argmax(), 1].item()
                    old_old = r[r[:, 0].argmax(), 2].item()
                    old_new = r[r[:, 0].argmax(), 3].item()
                    new_new = r[r[:, 0].argmax(), 4].item()
                    best_results.append((val, test_r, old_old, old_new, new_new))

                best_result = torch.tensor(best_results)

                r = best_result[:, 0]
                file.write(f'  Final val: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 1]
                file.write(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 2]
                file.write(f'   Final old_old: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 3]
                file.write(f'   Final old_new: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 4]
                file.write(f'   Final new_new: {r.mean():.2f} ± {r.std():.2f}\n')
    file.close()


if __name__ == "__main__":
    main()

