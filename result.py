import os
import torch
import json
import pandas

from utils import DotDict, Logger, rmse, rmse_tensor, boolean_string, get_dir, get_time, next_dir, get_model, model_dir

# print the information for the given model | *test_time* 

def get_config(model_dir, print_list = []):
    # get config
    with open(os.path.join(model_dir, 'config.json')) as f:
        config_logs = json.load(f)
    for opt in print_list:
        print(config_logs[opt])
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


def get_df(folder, col=['test_loss', 'nhid', 'nlayers'], required_list = 'all'):
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


class Printer():
    def __init__(self, folder):
        self.folder = folder
        self.dataset = self.folder.split('_')[0]
        self.model = self.folder.split('_')[1]
    def next_dir_list(self):
        return next_dir(folder)
    def get_list(self, string):
        model_list = next_dir(self.folder)
        li = []
        for i in model_list:
            if string in i:
                li.append(i)
        return li
    def get_df(self, col=['test_loss', 'nhid', 'nlayers'], required_list = 'all', mean=False, min=False):
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
    def min_idx(self, col=['test_loss', 'nhid', 'nlayers'], required_list = 'all'):
        df = self.get_df(col=col, required_list=required_list)
        print("the df is :")
        print(df)
        return df.idxmin()['test_loss']



# test
# folder = os.path.abspath(os.path.join(os.getcwd(), "..", "output", 'test'))
# info('D:/Jupyter_Documents/ML-code/research_code/output/aids_LSTM/test_21-53-28-11-20')

# df = get_df('D:/Jupyter_Documents/ML-code/research_code/output/aids_LSTM')
# df.mean()



# by list : 

# get_list('nhid_10', folder)
# print(get_list('00_08', folder))
# print(get_df(folder, required_list=get_list('test_00_08', folder)))   
# print(get_df(folder, required_list=[]))

# class test
# lstm = Printer(folder)
# print(lstm.get_df(required_list=lstm.get_list('10-40')))
# df = lstm.get_df(mean=True, min=True)
# lstm.,min_idx()
# df.idxmin()['test_loss']
# df.min()
# df.loc['row mean'] = df.apply(lambda x: x.mean())
# df
# predictions = {}
# for exp in exps:
#     model = models[exp]
#     model.eval()
#     with torch.no_grad():
#         prediction = model.generate(test_input, 396)
#         predictions[exp] = prediction
#         mse =  rmse(prediction, test_data)
#         print(prediction.size())
#     print('RMSE {}:/t'.format(exp), mse)

# plt.figure('Test plots', figsize=(17, 4), dpi=90)
# for exp in exps:
#     with open(os.path.join(outputdir, exp, 'logs.json'), 'r') as f:
#         logs = json.load(f)

#     plt.plot([logs['test_epoch.ts.{}.rmse'.format(ts)][-1] for ts in range(396)], label=exp, alpha=0.8)

# plt.grid()
# plt.title('Prediction RMSE')
# plt.xlabel('timestep')
# plt.legend()