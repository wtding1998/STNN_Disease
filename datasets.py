import os

import numpy as np
import torch

from utils import DotDict, normalize


def dataset_factory(data_dir, disease, nt_train, k=1):
    # get dataset
    opt, data, relations = get_data_set(data_dir, disease)
    # make k hop
    new_rels = [relations]
    for n in range(k - 1):
        new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
    relations = torch.cat(new_rels, 1)
    # split train / test
    train_data = data[:nt_train]
    test_data = data[nt_train:]
    return opt, (train_data, test_data), relations


def get_data_set(data_dir, disease):
    # dataset configuration
    opt = DotDict()
    opt.nt = 156
    opt.nx = 29
    opt.nd = 1
    opt.periode = opt.nt
    file = disease + ".csv"
    # loading data
    data = np.genfromtxt(os.path.join(data_dir, file), encoding="utf-8-sig", delimiter=",")
    # change data
    # data = change(data)
    data = torch.Tensor(data).view(opt.nt, opt.nx, opt.nd)
    # load relations
    reletions_file = disease + "_relations.csv"
    relations = torch.Tensor(np.genfromtxt(os.path.join(data_dir, 'flu_relations.csv'), encoding="utf-8-sig", delimiter=","))
    relations = normalize(relations).unsqueeze(1)
    return opt, data, relations

# def change(data):
#     print("max = %f" %np.max(data))
#     return data / np.max(data)

