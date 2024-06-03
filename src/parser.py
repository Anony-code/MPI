import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Graph Imputation using PyTorch and GNN')
    parser.add_argument('--num_features', type=int, default=128, help='Number of node features')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--fast_split', type=bool, default=False, help='Batch size for training')
    parser.add_argument('--use_valedges_as_input', type=bool, default=False, help='Batch size for training')
    parser.add_argument('--encoder', default='adj2gnn', type=str)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='ehr')
    parser.add_argument('--predictor', type=str, default='mlp', choices=['inner', 'mlp'])
    parser.add_argument('--patience', type=int, default=40, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@10', help='main evaluation metric')
    parser.add_argument('--transductive', type=str, default='transductive', choices=['transductive', 'production'])
    parser.add_argument('--minibatch', action='store_true')
    parser.add_argument('--task', type=str, default='ed')
    parser.add_argument('--scaled', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--msg_edge', type=str, default='ed_kd')
    parser.add_argument('--msg_edge_other', type=str, default='e-me')

    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--split_type', type=int, default=2)
    parser.add_argument('--rank', action='store_true')
    parser.add_argument('--margin', type=float, default=3.0)

    parser.add_argument('--self_loop', action='store_true')
    parser.add_argument('--cfg', type=list, default=[128, 64])

    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--withmap',  action='store_true')
    parser.add_argument('--code_init', action='store_true')

    parser.add_argument('--ratio', type=float, default=None)
    parser.add_argument('--allbest', action='store_true')
    parser.add_argument('--trans', type=str, default='p')

    parser.add_argument('--meta_card_book', type=int, default=96)
    parser.add_argument('--n_book', type=int, default=3)
    parser.add_argument('--prot_card_book', type=int, default=64)


    parser.add_argument('--runseed', type=int, default=0)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--dropadj',action='store_true')

    parser.add_argument('--fre_filter', type=int, default=1500)
    parser.add_argument('--kd', type=float, default=0.01)
    parser.add_argument('--kd_mse', type=float, default=1.0)

    args = parser.parse_args()
    return args

