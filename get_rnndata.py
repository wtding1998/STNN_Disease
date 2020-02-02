import os

import numpy as np
import torch

from utils import DotDict, normalize


def dataset_factory(data_dir, disease, nt_train, seq_len):
    # get dataset
    opt, data = get_data_set(data_dir, disease)
    # split train / test
    train_input = []
    train_output = []
    for i in range(nt_train - seq_len):
        new_input = []
        for j in range(seq_len):
            new_input.append(data[i+j])

        train_input.append(torch.stack(new_input, dim=0))
        train_output.append(data[i+seq_len])
    train_input = torch.stack(train_input, dim=0)
    train_output = torch.stack(train_output, dim=0)
    test_input = []
    for i in range(seq_len):
        test_input.append(data[nt_train-seq_len+i])
    test_data = data[nt_train:]
    test_input = torch.stack(test_input, dim=0)
    return opt, (train_input, train_output), (test_input, test_data.unsqueeze(2)) 


def get_data_set(data_dir, disease):
    # dataset configuration
    opt = DotDict()
    opt.nt = 156
    opt.nx = 29
    file = disease + ".csv"
    # loading data
    data = np.genfromtxt(os.path.join(data_dir, file), encoding="utf-8-sig", delimiter=",")
    # change data
    data = change(data)
    data = torch.Tensor(data).view(opt.nt, opt.nx)
    return opt, data

def change(data):
    return data / np.max(data)

if __name__ == "__main__":
    opt, data = get_data_set("data", "aids")
    print(data)
    