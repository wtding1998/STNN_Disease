import os
import torch
import json
import pandas
import numpy 
import matplotlib.pyplot as plt
from rnn_model import *
from utils import normalize, DotDict, Logger, rmse, rmse_tensor, boolean_string, get_dir, get_time, next_dir, get_model, model_dir
from stnn import SaptioTemporalNN

def get_config(model_dir):
    # get config
    with open(os.path.join(model_dir, 'config.json')) as f:
        config_logs = json.load(f)
    # for opt in print_list:
    #     print(config_logs[opt])
    # print("the test loss for %s is : %f" %(model_dir, config_logs['test_loss']))
    return config_logs

def get_logs(model_dir):
    # get logs
    with open(os.path.join(model_dir, 'logs.json')) as f:
        logs = json.load(f)
    return logs

# print the information for all model of the given folder | aids_LSTM

def get_list(string, folder):
    model_list = next_dir(folder)
    li = []
    for i in model_list:
        if string in i:
            li.append(i)
    return li


def get_df(folder, col=['test_loss', 'train_loss' 'nhid', 'nlayers'], required_list = 'all'):
    if isinstance(required_list, str):
        required_list = next_dir(folder)
    df_list = []
    for model_name in required_list: 
        config = get_config(os.path.join(folder, model_name))
        new_df = pandas.DataFrame([config])[col]
        new_df.index = [model_name]
        df_list.append(new_df)
    df =  pandas.concat(df_list, join='outer')
    df.name = folder.split('/')[-1]
    return df


class Exp():
    def __init__(self, exp_name, path):
        self.path = path
        print(self.path)
        self.exp_name = exp_name
        print(self.exp_name)
        self.config = get_config(os.path.join(self.path, self.exp_name))

    def model_name(self):
        folder_name = os.path.basename(os.path.normpath(self.path))
        return folder_name.split('_')[1]

    def dataset_name(self):
        folder_name = os.path.basename(os.path.normpath(self.path))
        return folder_name.split('_')[0]

    def get_relations(self):
        dataset = self.dataset_name()
        if dataset == "heat":
            data_path = os.path.abspath(os.path.join(self.path, "..", "..", "heat_STNN", "data"))
            relations = np.genfromtxt(os.path.join(data_path, "heat_relations.csv"), delimiter=',', encoding='utf-8-sig')
        else:
            data_path = os.path.abspath(os.path.join(self.path, "..", "..", "disease_STNN", "data"))
            relations = np.genfromtxt(os.path.join(data_path, dataset+"_relations.csv"), delimiter=',', encoding='utf-8-sig')
        relations = torch.tensor(relations).float()
        relations = normalize(relations).unsqueeze(1)
        return relations
        

    def logs(self):
        return get_logs(os.path.join(self.path, self.exp_name))

    def model(self):
        if self.model_name() == 'LSTM':
            model = LSTMNet(self.config['nx'], self.config['nhid'], self.config['nlayers'], self.config['nx'], self.config['seq_length'])
        if self.model_name() == 'GRU':
            model = GRUNet(self.config['nx'], self.config['nhid'], self.config['nlayers'], self.config['nx'], self.config['seq_length'])
        else:
            model = SaptioTemporalNN(self.get_relations(), self.config['nx'], self.config['checkpoint_interval'], 1, self.config['nz'], self.config['mode'], self.config['nhid'],
                             self.config['nlayers'], self.config['dropout_f'], self.config['dropout_d'],
                             self.config['activation'], self.config['periode'])
        model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt')))
        return model
    
    def pred(self, test_input=None, time=0):
        if os.path.exists(os.path.join(self.path, self.exp_name, 'pred.txt')):
            pred = np.genfromtxt(os.path.join(self.path, self.exp_name, 'pred.txt'))
        else:
            print('no pred.txt')
            pred = None
        return torch.tensor(pred)
          
class Printer():
    def __init__(self, folder):
        self.folder = folder
        self.dataset = self.folder.split('_')[0]
        self.model = self.folder.split('_')[1]

    def models(self):
        return next_dir(self.folder)

    def get_model(self, string):
        model_list = next_dir(self.folder)
        li = []
        for i in model_list:
            if string in i:
                li.append(i)
        return li

    def get_df(self, col=['test_loss', 'train_loss', 'nhid', 'nlayers'], required_list = 'all', mean=False, min=False):
        if isinstance(required_list, str):
            required_list = next_dir(self.folder)
        df_list = []
        for model_name in required_list: 
            config = get_config(os.path.join(self.folder, model_name))
            new_df = pandas.DataFrame([config])[col]
            new_df.index = [model_name]
            df_list.append(new_df)
        df =  pandas.concat(df_list, join='outer')
        if mean:
            df.loc['mean'] = df.apply(lambda x: x.mean())
        if min:
            df.loc['min'] = df.apply(lambda x: x.min())
        return df

    def min_idx(self, col=['test_loss', 'train_loss', 'nhid', 'nlayers'], required_list = 'all'):
        df = self.get_df(col=col, required_list=required_list)
        print("the df is :")
        print(df)
        return df.idxmin()['test_loss']
