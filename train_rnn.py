import os
import random
from collections import defaultdict, OrderedDict
import json
import datetime

import configargparse
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


from get_dataset import get_rnn_dataset
from utils import DotDict, Logger, rmse, rmse_tensor, boolean_string, get_dir, get_time, time_dir, shuffle_list
from rnn_model import LSTMNet, GRUNet, LSTMNet_onelinear, GRUNet_onelinear
import numpy as np


#######################################################################################################################
# Options - CUDA - Random seed
#######################################################################################################################
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')
p.add('--dataset', type=str, help='dataset name', default='ncov_confirmed')
p.add('--nt_train', type=int, help='time for training', default=50)
p.add('--start_time', type=int, help='time for training', default=0)
# -- xp
p.add('--outputdir', type=str, help='path to save xp', default='default')
p.add('--xp', type=str, help='xp name', default='rnn')
# p.add('--dir_auto', type=boolean_string, help='dataset_model', default=True)
p.add('--xp_auto', type=boolean_string, help='time', default=False)
p.add('--xp_time', type=boolean_string, help='xp_time', default=True)
p.add('--auto', type=boolean_string, help='dataset_model + time', default=False)
# -- model
p.add('--seq_length', type=int, help='sequence length', default=3)
p.add('--nhid', type=int, help='dynamic function hidden size', default=50)
p.add('--nlayers', type=int, help='dynamic function num layers', default=2)
p.add('--rnn_model', type=str, help='choose rnn model : LSTM | GRU', default='GRU')
# -- optim
p.add('--lr', type=float, help='learning rate', default=1e-2)
p.add('--sch_bound', type=float, help='bound for schedule', default=0.1)
p.add('--sch_factor', type=float, help='scheduler factor', default=0.9)
p.add('--clip_value', type=float, help='clip_value for learning', default=5.0)
p.add('--beta1', type=float, default=.9, help='adam beta1')
p.add('--beta2', type=float, default=.999, help='adam beta2')
p.add('--eps', type=float, default=1e-9, help='adam eps')
p.add('--wd', type=float, help='weight decay', default=1e-6)

# -- learning
p.add('--batch_size', type=int, default=10, help='batch size')
p.add('--patience', type=int, default=400, help='number of epoch to wait before trigerring lr decay')
p.add('--nepoch', type=int, default=20000, help='number of epochs to train for')
p.add('--test', type=boolean_string, default=False, help='test during training')
# -- gpu
p.add('--device', type=int, default=-1, help='-1: cpu; > -1: cuda device id')
# -- seed
p.add('--manualSeed', type=int, help='manual seed')
# -- logs
p.add('--checkpoint_interval', type=int, default=700, help='check point interval')
# parse
opt = DotDict(vars(p.parse_args()))

# cudnn
if opt.device > -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.device > -1:
    torch.cuda.manual_seed_all(opt.manualSeed)

#######################################################################################################################
# Data
#######################################################################################################################
# -- load data
setup, (train_input, train_output), (test_input, test_data) = get_rnn_dataset(opt.datadir, opt.dataset, opt.nt_train,opt.seq_length, opt.start_time)
train_input = train_input.to(device)
train_output = train_output.to(device)
test_input = test_input.to(device)
test_data = test_data.to(device)

for k, v in setup.items():
    opt[k] = v

if opt.outputdir == 'default':
    opt.outputdir = opt.dataset + "_" + opt.rnn_model
opt.outputdir = get_dir(opt.outputdir)

if opt.xp_time:
    opt.xp = opt.xp + "_" + get_time()
if opt.xp_auto:
    opt.xp = get_time()
if opt.auto_all:
    opt.outputdir = opt.dataset + "_" + opt.rnn_model 
    opt.xp = get_time()

opt.start = time_dir()
start_st = datetime.datetime.now()
opt.st = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
#######################################################################################################################
# Model
#######################################################################################################################
if opt.rnn_model == 'LSTM':
    model = LSTMNet(opt.nx * opt.nd, opt.nhid, opt.nlayers, opt.nx * opt.nd, opt.seq_length).to(device)
if opt.rnn_model == 'LSTM_one':
    model = LSTMNet_onelinear(opt.nx * opt.nd, opt.nhid, opt.nlayers, opt.nx * opt.nd, opt.seq_length).to(device)
if opt.rnn_model == 'GRU':
    model = GRUNet(opt.nx * opt.nd, opt.nhid, opt.nlayers, opt.nx * opt.nd, opt.seq_length).to(device)
if opt.rnn_model == 'GRU_one':
    model = GRUNet_onelinear(opt.nx * opt.nd, opt.nhid, opt.nlayers, opt.nx * opt.nd, opt.seq_length).to(device)


#######################################################################################################################
# Optimizer
#######################################################################################################################
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)
# optimizer = optim.SGD(model.parameters(), lr=opt.lr)
if opt.patience > 0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = opt.sch_factor, patience=opt.patience)
#######################################################################################################################
# Logs
#######################################################################################################################
logger = Logger(get_dir(opt.outputdir), opt.xp, opt.checkpoint_interval)
#######################################################################################################################
# Training
#######################################################################################################################
lr = opt.lr
pb = trange(opt.nepoch)
opt.mintest = 1000.0
for e in pb:
    # ------------------------ Train ------------------------
    batches = shuffle_list(opt.nt_train - opt.seq_length, opt.batch_size)
    for batch in batches:
        model.train()
        optimizer.zero_grad()
        prediction = model(train_input[batch])
        train_loss = rmse_tensor(train_output[batch], prediction)
        train_loss.backward()
    ## 按范数裁剪
    ### 这里norm_type可以选择L1范数，L2范数和无穷范数，分别对应`1, 2, 'inf'`
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_value, norm_type=2)

    ## 按值裁剪
    ### 指定clip_value之后，裁剪的范围就是[-clip_value, clip_value]
        # torch.nn.utils.clip_grad_value_(model.parameters(), opt.clip_value)
        # for name, param in model.named_parameters():
        #     print(name, param)
        optimizer.step()
        # logger.log('train_loss', train_loss.item())
    # checkpoint
    # logger.log('train_epoch.lr', lr)
    if opt.test:
        # ------------------------ Test ------------------------
        model.eval()
        with torch.no_grad():
            pred = model.generate(test_input, opt.nt - opt.nt_train)
            score = rmse(pred, test_data)
            # logger.log('test_epoch.rmse', score)
            pb.set_postfix(train_loss=train_loss.item(), test_loss=score)
            if opt.mintest > score:
                opt.mintest = score
            # schedule lr
            if opt.patience > 0 and score < opt.sch_bound:
                lr_scheduler.step(score)
            lr = optimizer.param_groups[0]['lr']
            if lr <= 1e-6:
                break
    else:
        pb.set_postfix(train_loss=train_loss.item())
# ------------------------ Test ------------------------
model.eval()
with torch.no_grad():
    pred = model.generate(test_input, opt.nt - opt.nt_train)
    pred = pred.view(opt.nt - opt.nt_train, opt.nx, opt.nd)
    test_data = test_data.view(opt.nt - opt.nt_train, opt.nx, opt.nd)
    score = rmse(pred, test_data)
    score_ts = rmse(pred, test_data, reduce=False) # 1-dim tensor
    if opt.normalize == 'max_min':
        pred = pred * (opt.max - opt.min) + opt.mean
        opt.true_loss = score * (opt.max - opt.min)
    if opt.normalize == 'variance':
        pred = pred * opt.std + opt.mean
        opt.true_loss = score * opt.std
    for i in range(opt.nd):
        d_pred = pred[:,:, i].cpu().numpy()
        np.savetxt(os.path.join(get_dir(opt.outputdir), opt.xp, 'pred_' + str(i).zfill(3) +  '.txt'), d_pred)
opt.test_loss = score
opt.train_loss = train_loss.item()
opt.end = time_dir()
end_st = datetime.datetime.now()
opt.et = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
opt.time = str(end_st - start_st)

with open(os.path.join(get_dir(opt.outputdir), opt.xp, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)

logger.save(model)

