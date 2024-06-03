import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, GINConv
from torch_geometric.nn.conv import RGCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, data_name, num_nodes, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, conv_layer, norm_type="none"):
        super(SAGE, self).__init__()

        self.node_embedding = nn.Embedding(num_nodes, in_channels)

        self.convs = torch.nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        elif self.norm_type == "layer":
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.convs.append(conv_layer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_channels))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_channels))
        self.convs.append(conv_layer(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.node_embedding.reset_parameters()

    def forward(self, x, adj_t):
        # import ipdb; ipdb.set_trace()

        if x.dtype == torch.int64:
            x = self.node_embedding(torch.arange(0, self.node_embedding.weight.shape[0], device=adj_t.device))

        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.norm_type != "none":
                    x = self.norms[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictorDot(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictorDot, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        elif self.predictor == 'inner':
            x = torch.sum(x, dim=-1)

        return torch.sigmoid(x)


class LinkPredictor(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.concat((x_i, x_j), dim=-1)
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.leaky_relu(x, 0.1)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        elif self.predictor == 'inner':
            x = torch.sum(x, dim=-1)

        return x


def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num = len(cfg)
    for i, v in enumerate(cfg):
        out_channels = v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.LeakyReLU(negative_slope=0.1)]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)

class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers, num_relation,
                 dropout):
        super(RGCN, self).__init__()
        self.node_embedding = nn.Embedding(num_nodes, in_channels)

        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relation))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations=num_relation))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations=num_relation))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.node_embedding.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        if x.dtype == torch.int64:
            x = self.node_embedding(torch.arange(0, self.node_embedding.weight.shape[0], device=edge_index.device))
            x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return x

class Adj2GNN(nn.Module):
    def __init__(self, num_nodes, in_channels, cfg=None, dropout=0.2):
        super(Adj2GNN, self).__init__()
        self.node_embedding = nn.Embedding(num_nodes, in_channels)

        self.MLP = make_mlplayers(in_channels, cfg)
        self.act = nn.ReLU()
        self.dropout = dropout
        self.A = None
        self.sparse = True

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        self.node_embedding.reset_parameters()

    def embed(self,  seq_a , adj=None ):
        h_a = self.MLP(seq_a)
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
        else:
            h_p = torch.mm(adj, h_a)
        return h_a.detach(), h_p.detach()

    def forward(self, seq_a, adj=None,  u_mul_s=None, vt=None):
        seq_a = self.node_embedding(torch.arange(0, self.node_embedding.weight.data.size(0), device=adj.device))
        if self.A is None:
            self.A = adj
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.MLP(seq_a)
        h_p_0 = F.dropout(h_a, self.dropout, training=self.training)

        if vt is not None:
            vt_ei = vt @ h_p_0
            G_svd = (u_mul_s @ vt_ei)
            return G_svd
        else:
            if self.sparse:
                h_p = torch.spmm(adj,  torch.spmm(adj, h_p_0))
                # print(h_p.shape, h_p)
            else:
                h_p = torch.mm(adj, h_p_0)
        return h_p


class Adj2GNNinit(nn.Module):
    def __init__(self, num_nodes, in_channels, in_channels2, init_fea1, init_fea2, cfg=None, dropout=0.2):
        super(Adj2GNNinit, self).__init__()
        self.node_embedding = nn.Embedding(num_nodes, in_channels)

        self.MLP = make_mlplayers(in_channels, cfg)
        self.act = nn.ReLU()

        self.dropout = dropout
        self.A = None
        self.sparse = True
        self.code_map = nn.Linear(in_channels2, in_channels)
        self.init1 = init_fea1
        self.init2 = init_fea2

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        self.node_embedding.reset_parameters()
        self.code_map.reset_parameters()

    def embed(self,  seq_a , adj=None ):
        h_a = self.MLP(seq_a)
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
        else:
            h_p = torch.mm(adj, h_a)
        return h_a.detach(), h_p.detach()

    def forward(self, seq_a, adj=None,  u_mul_s=None, vt=None):
        seq_a_p = self.node_embedding(torch.arange(0, self.node_embedding.weight.data.size(0), device=adj.device))
        seq_a_p = F.dropout(seq_a_p, self.dropout, training=self.training)

        if self.init1 is None:
            seq_a_code = self.init2

        elif self.init2 is None:
            seq_a_code = self.init1
        else:
            seq_a_code = torch.cat([self.init1, self.init2], dim=0)

        seq_a_code_m = self.code_map(seq_a_code)
        seq_a_code_m = F.dropout(seq_a_code_m, self.dropout, training=self.training)

        seq_a = torch.cat([seq_a_p, seq_a_code_m], dim=0)
        if self.A is None:
            self.A = adj

        h_a = self.MLP(seq_a)
        h_p_0 = F.dropout(h_a, self.dropout, training=self.training)
        if vt is not None:
            vt_ei = vt @ h_p_0
            G_svd = (u_mul_s @ vt_ei)
            return G_svd
        else:
            if self.sparse:
                h_p = torch.spmm(adj,  torch.spmm(adj, h_p_0))
            else:
                h_p = torch.mm(adj, h_p_0)
        return h_p
    
class GIN(torch.nn.Module):
    def __init__(self, data_name, num_nodes, in_channels, hidden_channels, out_channels, dropout):
        super(GIN, self).__init__()

        self.node_embedding = nn.Embedding(num_nodes, in_channels)

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, out_channels),
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        self.convs = [self.conv1, self.conv2]
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.node_embedding.reset_parameters()

    def forward(self, x, adj_t):

        if x.dtype == torch.int64:
            x = self.node_embedding(torch.arange(0, self.node_embedding.weight.shape[0], device=adj_t.device))

        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x