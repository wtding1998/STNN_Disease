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

# TODO: ADD A CLASS
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



# test
# info('D:/Jupyter_Documents/ML-code/research_code/output/aids_LSTM/test_21-53-28-11-20')

# df = get_df('D:/Jupyter_Documents/ML-code/research_code/output/aids_LSTM')
# df.mean()

# by list : 
# folder = os.path.abspath(os.path.join(os.getcwd(), "..", "output", 'aids_LSTM'))
# get_list('nhid_10', folder)
# print(get_list('00_08', folder))
# print(get_df(folder, required_list=get_list('test_00_08', folder)))   
# print(get_df(folder, required_list=[]))

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