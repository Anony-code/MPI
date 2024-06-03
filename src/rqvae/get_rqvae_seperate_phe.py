import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import numpy as np
from torch.utils.data import DataLoader
from ema import ExponentialMovingAverage
import os
import torch
import torch.optim as optim
import pickle

def create_model(input_dim,n_embed, config, ema=False):
    from rqvae import RQVAE

    model_type = 'rq-vae'
    if model_type == 'rq-vae':
        model = RQVAE(input_dim, n_embed,   kwargs= config)
        model_ema = RQVAE(input_dim, n_embed,  kwargs=config) if ema else None
    else:
        raise ValueError(f'{model_type} is invalid..')

    if ema:
        model_ema = ExponentialMovingAverage(model_ema, 0.3)
        model_ema.eval()
        model_ema.update(model, step=-1)

    return model, model_ema


def create_two_head_model(input_dim, input_dim2, hid_dim, n_embed, config, ema=False):
    from rqvae import RQVAE

    model_type = 'rq-vae'
    if model_type == 'rq-vae':
        model = RQVAE(input_dim, input_dim2, hid_dim,  n_embed, kwargs= config, loss_type='l1')
        model_ema = RQVAE(input_dim, input_dim2,  hid_dim, n_embed, kwargs=config, loss_type='l1') if ema else None
    else:
        raise ValueError(f'{model_type} is invalid..')

    if ema:
        model_ema = ExponentialMovingAverage(model_ema, 0.3)
        model_ema.eval()
        model_ema.update(model, step=-1)

    return model, model_ema


def create_one_head_model(input_dim, input_dim2, hid_dim, n_embed, config, ema=False, loss_type=None, lat=None):
    from rqvae import RQVAE
    assert input_dim2 is None
    model_type = 'rq-vae'
    if model_type == 'rq-vae':
        model = RQVAE(input_dim, None, hid_dim,  n_embed, kwargs= config, loss_type=loss_type, latent_loss_weight=lat)
        model_ema = RQVAE(input_dim, None,  hid_dim, n_embed, kwargs=config, loss_type=loss_type, latent_loss_weight=lat) if ema else None
    else:
        raise ValueError(f'{model_type} is invalid..')

    if ema:
        model_ema = ExponentialMovingAverage(model_ema, 0.3)
        model_ema.eval()
        model_ema.update(model, step=-1)

    return model, model_ema

file_attr = ['../../pkl/attr_matrix_unchanged_ep_phe.pkl', '../../pkl/attr_matrix_zscore_em_phe.pkl']
file_row = ['../../pkl/attr_node_list_row_p_phe.pkl', '../../pkl/attr_node_list_row_m_phe.pkl']

omics = 'ep'
file_attr = file_attr[:1]
file_row = file_row[:1]

attr_matrix = []
for f, frow in zip(file_attr, file_row):
    random_tensor = pickle.load(open(f, 'rb'))  # 15339, 251
    mask_row = random_tensor.sum(axis=-1) != 0

    random_tensor = random_tensor[mask_row]
    attr_matrix.append(torch.tensor(random_tensor).float())

kwargs = {}
kwargs['latent_shape'] = [32, 1]
kwargs['code_shape'] = [32, 3]
kwargs['shared_codebook'] = False
kwargs['restart_unused_codes'] = True
kwargs['n_book'] = 3
kwargs['card_book'] = 64
kwargs['lr'] = 0.0005
kwargs['lat'] = 1.0


model, model_ema = create_one_head_model(attr_matrix[0].shape[-1], None,
                                         kwargs['latent_shape'][0], kwargs['card_book'], kwargs, loss_type='mse', lat=kwargs['lat'])

optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])  # Change learning rate if needed

best_loss = np.inf
epoch = 100000
batch_size = 512

train_val_perm = torch.randperm(attr_matrix[0].size(0))
train_perm, val_perm = train_val_perm[:int(attr_matrix[0].size(0) * 0.8)],  train_val_perm[int(attr_matrix[0].size(0) * 0.8):]
attr_matrix1 = attr_matrix[0]

if omics == 'ep':
    attr_matrix1_np = attr_matrix1.numpy()
    column_means = np.nanmean(attr_matrix1_np, axis=0)
    nan_indices = np.where(np.isnan(attr_matrix1_np))

    attr_matrix1_np[nan_indices] = np.take(column_means, nan_indices[1])

    attr_matrix1 = torch.tensor(attr_matrix1_np).to(torch.float)
attr_matrix1_train = attr_matrix1.index_select(0, train_perm)
attr_matrix1_val = attr_matrix1.index_select(0, val_perm)


def val_step(model, val_data, batch_size):
    loss_rec_lat_avg = 0
    loss_recon_avg = 0
    loss_latent_avg = 0
    loader = DataLoader(range(val_data.size(0)), batch_size=32, shuffle=False)
    with torch.no_grad():
        for i, perm in enumerate(loader):
            # optimizer.zero_grad(set_to_none=True)
            xs = val_data[perm]
            outputs = model(xs, flag=1)
            xs_recon = outputs[0]

            outputs = model.compute_loss(*outputs, xs=xs)  # the recons loss

            loss_rec_lat = outputs['loss_total']
            loss_recon = outputs['loss_recon']
            loss_latent = outputs['loss_latent']
            loss_rec_lat_avg += loss_rec_lat.item()
            loss_recon_avg += loss_recon.item()
            loss_latent_avg += loss_latent.item()
    loss_rec_lat_avg = loss_rec_lat_avg / len(loader)
    loss_recon_avg = loss_recon_avg / len(loader)
    loss_latent_avg = loss_latent_avg / len(loader)

    return loss_rec_lat_avg, loss_recon_avg, loss_latent_avg

cnt_wait = 0
best_ej = 0
for ej in range(epoch):

    for i, perm in enumerate(DataLoader(range(attr_matrix1_train.size(0)), batch_size=batch_size, shuffle=True)):
        optimizer.zero_grad(set_to_none=True)
        xs = attr_matrix1_train[perm]
        outputs = model(xs, flag=1)

        xs_recon = outputs[0]
        outputs = model.compute_loss(*outputs, xs=xs)  # the recons loss

        loss_rec_lat = outputs['loss_total']
        loss_recon = outputs['loss_recon']
        loss_latent = outputs['loss_latent']

        curLoss = loss_rec_lat  #+ loss_rec_lat2
        curLoss.backward()
        optimizer.step()

    valloss_all, valloss_rec, valloss_lat = val_step(model, attr_matrix1_val, batch_size)

    if valloss_all <= best_loss:
        best_loss = valloss_all
        cnt_wait = 0
        if best_ej < ej:
            os.remove('rqckpt/rqvae_model_parameters_{}_{}.pth'.format(omics, best_ej))

        best_ej = ej
        torch.save(model.state_dict(), 'rqckpt/rqvae_model_parameters_{}_{}.pth'.format(omics, best_ej))

    else:
        cnt_wait += 1

    if cnt_wait >= 500: 
        break


model.load_state_dict(torch.load('rqckpt/rqvae_model_parameters_{}_{}.pth'.format(omics, best_ej), map_location=torch.device('cpu')))

outputs1 = model(attr_matrix1, flag=1)
outputs1 = model.compute_loss(*outputs1, xs=attr_matrix1) # the recons loss

all_codes1 = outputs1['codes']
all_codes1 = torch.concat(all_codes1, dim=0)

code_set1 = all_codes1[:, 0].detach().cpu().numpy()
code_set2 = all_codes1[:, 1].detach().cpu().numpy()
code_set3 = all_codes1[:, 2].detach().cpu().numpy()

from collections import Counter
# generate codes based on the rqvae 
print(Counter(list(code_set1)), Counter(list(code_set2)), Counter(list(code_set3)))
pickle.dump(all_codes1, open('rqpkl/sep_codes_ep_{}_{}_phe.pkl'.format(str(kwargs['n_book']), str(kwargs['card_book'])), 'wb'))

codebook_emb = model.quantizer.codebooks
book_emb = []
for book_ in codebook_emb:
    book_emb.append(book_.weight[:-1, :])
book_emb = torch.cat(book_emb, dim=0)
pickle.dump(book_emb, open('rqpkl/sep_codes_embs_ep_{}_{}_phe.pkl'.format(str(kwargs['n_book']), str(kwargs['card_book'])), 'wb'))

