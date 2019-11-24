from utils import *
from get_data import *
import os
import torch
from rnn_model import *
from result import *
import datetime
import time


# os test

# outputdir = os.path.abspath(os.path.join(os.getcwd(), "..", "output"))
# mode_dir = next_dir(outputdir)
# model_dir = {}
# path_dir = {}
# for mode in mode_dir:
#     di = os.path.join(outputdir, mode)
#     model_list = next_dir(di)  
#     model_dir[mode] = model_list
# print(model_dir)

# # load model test
# folder = 'D:/Jupyter_Documents/ML-code/research_code/output/aids_LSTM'
# lstm = Printer(folder)
# model_config = lstm.config('test_23-25-21-11-33')
# setup, (train_input, train_output), (test_input, test_data) = dataset_factory(model_config['datadir'], model_config['dataset'], model_config['nt_train'],model_config['seq_length'])
# # load
# model = LSTMNet(model_config['nx'], model_config['nhid'], model_config['nlayers'], model_config['nx'], model_config['seq_length'])
# model.load_state_dict(torch.load('D:/Jupyter_Documents/ML-code/research_code/output/aids_LSTM/test_23-25-21-11-33/model.pt'))
# pred = model.generate(test_input, model_config['nt'] - model_config['nt_train'])
# print(pred)
start = datetime.datetime.now()
time.sleep(1)
end = datetime.datetime.now()
print(end-start)
