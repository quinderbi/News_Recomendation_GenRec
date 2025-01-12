
import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from Config import Config

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
from DataProcessor import DataProcessor
from copy import deepcopy
import pandas as pd

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

config = Config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

ds_name = "ebnerd"

print("Loading dataset {}..." .format(ds_name))
train_df = pd.read_csv(f"data/{ds_name}/train_df.csv")
valid_df = pd.read_csv(f"data/{ds_name}/valid_df.csv")
test_df = pd.read_csv(f"data/{ds_name}/test_df.csv")

train_df.rename(columns={"user_id": "user", "item_id": "item"}, inplace=True)
valid_df.rename(columns={"user_id": "user", "item_id": "item"}, inplace=True)
test_df.rename(columns={"user_id": "user", "item_id": "item"}, inplace=True)
train_df['rating'] = 1
valid_df['rating'] = 1
test_df['rating'] = 1

ratings = pd.concat([train_df, valid_df, test_df], axis=0)

train_data = DataProcessor.construct_one_valued_matrix(ratings, train_df, item_based=False)
valid_y_data = DataProcessor.construct_one_valued_matrix(ratings, valid_df, item_based=False)
test_y_data = DataProcessor.construct_one_valued_matrix(ratings, test_df, item_based=False)

train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)

if True:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=config['batch_size'], shuffle=False)
mask_tv = train_data + valid_y_data

print('data ready.')

list_epoch = [50, 100]
best_score = 0.0
best_params = None
best_evals = None
best_model = None


for lr in [1e-6, 1e-5, 1e-4]:
    for batch_size in [64, 128]:

        config['lr'] = lr
        config['batch_size'] = batch_size

        ### Build Gaussian Diffusion ###
        if config['mean_type'] == 'x0':
            mean_type = gd.ModelMeanType.START_X
        elif config['mean_type'] == 'eps':
            mean_type = gd.ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % config['mean_type'])

        diffusion = gd.GaussianDiffusion(mean_type,config['noise_schedule'], \
                config['noise_scale'], config['noise_min'], config['noise_max'],config['steps'], device).to(device)

        ### Build MLP ###
        out_dims = config['dims'] + [ratings.item.nunique()]
        in_dims = out_dims[::-1]
        model = DNN(in_dims, out_dims, config['emb_size'], time_type=config['time_type'], norm=config['norm']).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        print("models ready.")

        def evaluate(data_loader, data_te, mask_his, topN):
            model.eval()
            e_idxlist = list(range(mask_his.shape[0]))
            e_N = mask_his.shape[0]

            predict_items = []
            target_items = []
            for i in range(e_N):
                target_items.append(data_te[i, :].nonzero()[1].tolist())
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    his_data = mask_his[e_idxlist[batch_idx*config['batch_size']:batch_idx*config['batch_size']+len(batch)]]
                    batch = batch.to(device)
                    prediction = diffusion.p_sample(model, batch, config['sampling_steps'], config['sampling_noise'])
                    prediction[his_data.nonzero()] = -np.inf

                    _, indices = torch.topk(prediction, topN[-1])
                    indices = indices.cpu().numpy().tolist()
                    predict_items.extend(indices)

            test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

            return test_results


        print("Start training...")
        for epoch in range(1, max(list_epoch) + 1):

            model.train()
            start_time = time.time()

            batch_count = 0
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                batch_count += 1
                optimizer.zero_grad()
                losses = diffusion.training_losses(model, batch, config['reweight'])
                loss = losses["loss"].mean()
                total_loss += loss
                loss.backward()
                optimizer.step()

            if epoch in list_epoch:
                valid_results = evaluate(test_loader, valid_y_data, train_data, [5, 10, 15, 20])
                score = np.mean(valid_results)
                print(valid_results)
                print("Score: ", score)
                if score > best_score:
                    best_score = score
                    best_params = (lr, batch_size, epoch)
                    best_evals = valid_results
                    best_model = deepcopy(model)


print("Best parameters: lr={}, batch_size={}, epoch={}".format(*best_params))
print("Best evaluation: ", best_evals)
print("Best score: ", best_score)

def evaluate(data_loader, data_te, mask_his, topN):
    best_model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*config['batch_size']:batch_idx*config['batch_size']+len(batch)]]
            batch = batch.to(device)
            prediction = diffusion.p_sample(best_model, batch, config['sampling_steps'], config['sampling_noise'])
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

test_results = evaluate(test_twv_loader, test_y_data, mask_tv, [5, 10, 15, 20])
print(test_results)
print("Score: ", np.mean(test_results))

with open(f"Diffusion_{ds_name}_results.txt", "w") as f:
    f.write("Best parameters: lr={}, batch_size={}, epoch={}\n".format(*best_params))
    f.write("Best evaluation: {}\n".format(best_evals))
    f.write("Best score: {}\n".format(best_score))
    f.write("\nEvaluate on test set:\n")
    f.write(str(test_results))
    f.write("\nScore: {}\n".format(np.mean(test_results)))

        




