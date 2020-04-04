import os
import random
from collections import defaultdict, OrderedDict
import json
import datetime

import configargparse
from tqdm import trange
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.optimizers import SGD, RMSprop, adam
import numpy as np

from get_dataset import get_keras_dataset
from utils import DotDict, Logger_keras, boolean_string, get_dir, get_time, time_dir, shuffle_list, rmse_np



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
p.add('--dropout', type=float, help='dropout rate', default=0.5)
p.add('--rnn_model', type=str, help='choose rnn model : LSTM | GRU', default='GRU')
# -- optim
p.add('--lr', type=float, help='learning rate', default=1e-2)
p.add('--validation_ratio', type=float, help='validation rate', default=0.1)
p.add('--clip_value', type=float, help='clip_value for learning', default=5.0)

# -- learning
p.add('--batch_size', type=int, default=10, help='batch size')
p.add('--nepoch', type=int, default=20000, help='number of epochs to train for')
# -- seed
p.add('--manualSeed', type=int, help='manual seed')
# -- logs
p.add('--checkpoint_interval', type=int, default=700, help='check point interval')
# parse
opt = DotDict(vars(p.parse_args()))


# seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)


#######################################################################################################################
# Data
#######################################################################################################################
# -- load data
setup, (train_input, train_output, test_input, test_data)= get_keras_dataset(opt.datadir, opt.dataset, opt.nt_train,opt.seq_length, opt.start_time)

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
opt.xp = 'keras-' + opt.xp
opt.start = time_dir()
start_st = datetime.datetime.now()
opt.st = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
#######################################################################################################################
# Model
#######################################################################################################################
model = Sequential()
# 1st layer
if opt.rnn_model == 'LSTM':
        model.add(LSTM(
            opt.nhid,
            input_shape=(None, opt.nx*opt.nd),
            return_sequences=True))
elif opt.rnn_model == 'GRU':
        model.add(GRU(
            opt.nhid,
            input_shape=(None, opt.nx*opt.nd),
            return_sequences=True))

model.add(Dropout(opt.dropout))
# middle layers
for i in range(opt.nlayers-2):
    if opt.rnn_model == 'LSTM':
        model.add(LSTM(
            opt.nhid,
            return_sequences=True))
    elif opt.rnn_model == 'GRU':
        model.add(GRU(
            opt.nhid,
            return_sequences=True))
    model.add(Dropout(opt.dropout))

# final layer
if opt.rnn_model == 'LSTM':
        model.add(LSTM(
            opt.nhid,
            return_sequences=False))
elif opt.rnn_model == 'GRU':
        model.add(GRU(
            opt.nhid,
            return_sequences=False))
model.add(Dropout(opt.dropout))

model.add(Dense(
    opt.nx*opt.nd))
model.add(Activation("tanh"))
model.compile(loss="mse", optimizer='rmsprop', metrics=['mae', 'mape'])

#######################################################################################################################
# Logs
#######################################################################################################################
logger = Logger_keras(get_dir(opt.outputdir), opt.xp, opt.checkpoint_interval)
#######################################################################################################################
# Training
#######################################################################################################################
model_history = model.fit(
    train_input, train_output,
    batch_size=opt.batch_size, epochs=opt.nepoch, validation_split=opt.validation_ratio)

#######################################################################################################################
# Test
#######################################################################################################################
# generate pred
pred = []
last_sequence = test_input[np.newaxis, ...]
for i in range(opt.nt - opt.nt_train):
    new_pred = model.predict(last_sequence)
    pred.append(new_pred)
    new_pred = new_pred[np.newaxis, ...]
    last_sequence = np.concatenate([last_sequence[:, 1:, :], new_pred], axis=1)
pred = np.concatenate(pred, axis=0)
pred = np.reshape(pred, (opt.nt - opt.nt_train, opt.nx, opt.nd))
test_data = np.reshape(test_data, (opt.nt - opt.nt_train, opt.nx, opt.nd))
# print(pred)
score = rmse_np(pred, test_data)
if opt.normalize == 'max_min':
    pred = pred * (opt.max - opt.min) + opt.mean
    opt.true_loss = score * (opt.max - opt.min)
if opt.normalize == 'variance':
    pred = pred * opt.std + opt.mean
    opt.true_loss = score * opt.std
opt.test_loss = score
train_loss_history = model_history.history['loss']
logger.log('train_loss', train_loss_history)
opt.train_loss = train_loss_history[-1]
opt.end = time_dir()
end_st = datetime.datetime.now()
opt.et = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
opt.time = str(end_st - start_st)

with open(os.path.join(get_dir(opt.outputdir), opt.xp, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)

logger.save(model)

for i in range(opt.nd):
    d_pred = pred[:,:, i]
    np.savetxt(os.path.join(get_dir(opt.outputdir), opt.xp, 'pred_' + str(i).zfill(3) +  '.txt'), d_pred)
