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


from get_rnndata import dataset_factory
from utils import DotDict, Logger, rmse, rmse_tensor, boolean_string, get_dir, get_time, time_dir
from rnn_model import LSTMNet, GRUNet
import numpy as np


#######################################################################################################################
# Options - CUDA - Random seed
#######################################################################################################################
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')
p.add('--dataset', type=str, help='dataset name', default='aids')
p.add('--nt_train', type=int, help='time for training', default=100)
# -- xp
p.add('--outputdir', type=str, help='path to save xp', default='output')
p.add('--xp', type=str, help='xp name', default='stnn')
p.add('--dir_auto', type=boolean_string, help='dataset_model', default=True)
p.add('--xp_auto', type=boolean_string, help='time', default=False)
p.add('--xp_time', type=boolean_string, help='xp_time', default=True)
p.add('--auto', type=boolean_string, help='dataset_model + time', default=False)
# -- model
p.add('--seq_length', type=int, help='sequence length', default=5)
p.add('--nhid', type=int, help='dynamic function hidden size', default=50)
p.add('--nlayers', type=int, help='dynamic function num layers', default=3)
p.add('--rnn_model', type=str, help='choose rnn model : LSTM | GRU', default='LSTM')
# -- optim
p.add('--lr', type=float, help='learning rate', default=3e-3)
p.add('--beta1', type=float, default=.9, help='adam beta1')
p.add('--beta2', type=float, default=.999, help='adam beta2')
p.add('--eps', type=float, default=1e-9, help='adam eps')
p.add('--wd', type=float, help='weight decay', default=1e-6)
# -- learning
p.add('--batch_size', type=int, default=10, help='batch size')
p.add('--patience', type=int, default=150, help='number of epoch to wait before trigerring lr decay')
p.add('--nepoch', type=int, default=100, help='number of epochs to train for')
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
setup, (train_input, train_output), (test_input, test_data) = dataset_factory(opt.datadir, opt.dataset, opt.nt_train,opt.seq_length)
train_input = train_input.to(device)
train_output = train_output.to(device)
test_input = test_input.to(device)
test_data = test_data.to(device)

for k, v in setup.items():
    opt[k] = v

if opt.dir_auto:
    opt.outputdir = opt.dataset + "_" + opt.rnn_model 
if opt.xp_time:
    opt.xp = opt.xp + "_" + get_time()
if opt.xp_auto:
    opt.xp = get_time()
if opt.auto_all:
    opt.outputdir = opt.dataset + "_" + opt.rnn_model 
    opt.xp = get_time()

opt.start = time_dir()
start_st = datetime.datetime.now()
opt.start_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
#######################################################################################################################
# Model
#######################################################################################################################
if opt.rnn_model == 'LSTM':
    model = LSTMNet(opt.nx, opt.nhid, opt.nlayers, opt.nx, opt.seq_length).to(device)
if opt.rnn_model == 'GRU':
    model = GRUNet(opt.nx, opt.nhid, opt.nlayers, opt.nx, opt.seq_length).to(device)


#######################################################################################################################
# Optimizer
#######################################################################################################################
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)
if opt.patience > 0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience)
#######################################################################################################################
# Logs
#######################################################################################################################
logger = Logger(get_dir(opt.outputdir), opt.xp, opt.checkpoint_interval)
with open(os.path.join(get_dir(opt.outputdir), opt.xp, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)
#######################################################################################################################
# Training
#######################################################################################################################
lr = opt.lr
pb = trange(opt.nepoch)
for e in pb:
    # ------------------------ Train ------------------------
    model.train()
    optimizer.zero_grad()
    prediction = model(train_input)
    train_loss = rmse_tensor(train_output, prediction)
    train_loss.backward()
    optimizer.step()
    logger.log('train_loss', train_loss.item())
    # checkpoint
    # logger.log('train_epoch.lr', lr)
    logger.checkpoint(model)
    if opt.test:
        # ------------------------ Test ------------------------
        model.eval()
        with torch.no_grad():
            pred = model.generate(test_input, opt.nt - opt.nt_train)
            score = rmse(pred, test_data)
            logger.log('test_epoch.rmse', score)
            pb.set_postfix(train_loss=train_loss.item(), test_loss=score)
    else:
        pb.set_postfix(train_loss=train_loss.item())
# ------------------------ Test ------------------------
model.eval()
with torch.no_grad():
    pred = model.generate(test_input, opt.nt - opt.nt_train)
    pred = pred.view(opt.nt - opt.nt_train, opt.nx, 1)
    test_data = test_data.view(opt.nt - opt.nt_train, opt.nx, 1)
    score = rmse(pred, test_data)
    score_ts = rmse(pred, test_data, reduce=False) # 1-dim tensor
    print("test_loss : %f" %score)
    pred = pred.view(opt.nt - opt.nt_train, opt.nx)
    pred = pred.cpu().numpy()
    np.savetxt(os.path.join(get_dir(opt.outputdir), opt.xp, 'pred.txt'), pred)
    logger.log('test_epoch.rmse', score)
    logger.log('test_epoch.ts', {t: {'rmse': scr.item()} for t, scr in enumerate(score_ts)}) # t : time, 0-55 scr : score
    # schedule lr
    # if opt.patience > 0 and score < 1:
    #     lr_scheduler.step(score)
    # lr = optimizer.param_groups[0]['lr']
    # if lr <= 1e-5:
    #     break
opt.test_loss = score
opt.train_loss = train_loss.item()
opt.end = time_dir()
end_st = datetime.datetime.now()
opt.end_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
opt.time = str(end_st - start_st)

with open(os.path.join(get_dir(opt.outputdir), opt.xp, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)
logger.save(model)
