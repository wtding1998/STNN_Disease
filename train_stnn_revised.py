'''
province order :
['上海市', '云南省', '内蒙古', '北京市', '吉林省', '四川省', '天津市', '宁夏', '安徽省', '山东省',
       '山西省', '广东省', '广西', '新疆', '江苏省', '江西省', '河北省', '河南省', '浙江省', '海南省',
       '湖北省', '湖南省', '甘肃省', '福建省', '西藏', '贵州省', '辽宁省', '重庆市', '陕西省', '青海省',
       '黑龙江省']
'''



import os
import random
import json
from collections import defaultdict, OrderedDict
import datetime
import numpy as np

import configargparse
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


from get_dataset import get_stnn_data
from utils import DotDict, Logger, rmse, boolean_string, get_dir, get_time, time_dir
from stnn import SaptioTemporalNN_tanh

def train(command=False):
    if command == True:
        #######################################################################################################################
        # Options - CUDA - Random seed
        #######################################################################################################################
        p = configargparse.ArgParser()
        # -- data
        p.add('--datadir', type=str, help='path to dataset', default='data')
        p.add('--dataset', type=str, help='dataset name', default='ncov_confirmed')
        p.add('--nt_train', type=int, help='time for training', default=50)
        p.add('--validation_ratio', type=float, help='validation/train', default=0.1)
        p.add('--start_time', type=int, help='start time for data', default=0)
        p.add('--rescaled', type=str, help='rescaled method', default='d')
        p.add('--normalize_method', type=str, help='normalize method for relation', default='all')
        p.add('--normalize', type=str, help='normalize method for time data (variance | min_max)', default='variance')

        # -- xp
        p.add('--outputdir', type=str, help='path to save xp', default='default')
        p.add('--xp', type=str, help='xp name', default='stnn')
        # p.add('--dir_auto', type=boolean_string, help='dataset_model', default=True)
        p.add('--xp_auto', type=boolean_string, help='time', default=False)
        p.add('--xp_time', type=boolean_string, help='xp_time', default=True)
        p.add('--auto', type=boolean_string, help='dataset_model + time', default=False)
        # -- model
        p.add('--model', type=str, help='STNN Model', default='default')
        p.add('--mode', type=str, help='STNN mode (default|refine|discover)', default='default')
        p.add('--nz', type=int, help='laten factors size', default=1)
        p.add('--activation', type=str, help='dynamic module activation function (identity|tanh)', default='tanh')
        p.add('--khop', type=int, help='spatial depedencies order', default=1)
        p.add('--nhid', type=int, help='dynamic function hidden size', default=0)
        p.add('--nlayers', type=int, help='dynamic function num layers', default=1)
        p.add('--nhid_de', type=int, help='dynamic function hidden size', default=0)
        p.add('--nlayers_de', type=int, help='dynamic function num layers', default=1)
        p.add('--dropout_f', type=float, help='latent factors dropout', default=.5)
        p.add('--dropout_d', type=float, help='dynamic function dropout', default=.5)
        p.add('--dropout_de', type=float, help='dynamic function dropout', default=.5)
        p.add('--lambd', type=float, help='lambda between reconstruction and dynamic losses', default=.1)
        # -- optim
        p.add('--lr', type=float, help='learning rate', default=3e-3)
        p.add('--optimizer', type=str, help='learning algorithm', default='Adam')
        p.add('--beta1', type=float, default=.0, help='adam beta1')
        p.add('--beta2', type=float, default=.999, help='adam beta2')
        p.add('--eps', type=float, default=1e-9, help='adam eps')
        p.add('--wd', type=float, help='weight decay', default=1e-6)
        p.add('--wd_z', type=float, help='weight decay on latent factors', default=1e-7)
        p.add('--l2_z', type=float, help='l2 between consecutives latent factors', default=0.)
        p.add('--l1_rel', type=float, help='l1 regularization on relation discovery mode', default=0.)
        p.add('--sch_bound', type=float, help='learning rate', default=0.001)
        # -- learning
        p.add('--batch_size', type=int, default=1131, help='batch size')
        p.add('--patience', type=int, default=150, help='number of epoch to wait before trigerring lr decay')
        p.add('--nepoch', type=int, default=10, help='number of epochs to train for')
        p.add('--test', type=boolean_string, default=False, help='test during training')

        # -- gpu
        p.add('--device', type=int, default=-1, help='-1: cpu; > -1: cuda device id')
        # -- seed
        p.add('--manualSeed', type=int, help='manual seed')
        # -- logs
        p.add('--checkpoint_interval', type=int, default=100, help='check point interval')

        # parse
        opt=DotDict(vars(p.parse_args()))
        
    else:
        print('Use Matlab')
        opt = DotDict()
        # -- data
        opt.datadir = 'data'
        opt.dataset = 'ncov_confirmed'
        opt.nt_train = 15
        opt.start_time = 0
        opt.rescaled = 'd'
        opt.normalize_method = 'row'
        # -- xp
        opt.outputdir = 'default'
        opt.xp = 'stnn'
        # opt.dir_auto =  True
        opt.xp_auto =  False
        opt.xp_time =  True
        opt.auto = False
        # -- model
        opt.mode = 'default'
        opt.nz =1
        opt.activation = 'tanh'
        opt.khop = 1
        opt.nhid = 0
        opt.nlayers =1
        opt.dropout_f = .5
        opt.dropout_d = .5
        opt.lambd = .1
        # -- optim
        opt.lr = 3e-3
        opt.beta1 = .0
        opt.beta2 = .999
        opt.eps = 1e-9 
        opt.wd = 1e-6
        opt.wd_z = 1e-7
        opt.l2_z = 0.
        opt.l1_rel = 0.
        opt.sch_bound = 0.017
        # -- learning
        opt.batch_size = 1000
        opt.patience = 150
        opt.nepoch = 100
        opt.test = False
        opt.device = -1
        print(opt)

    # if opt.dir_auto:
    #     opt.outputdir = opt.dataset + "_" + opt.mode 
    if opt.outputdir == 'default':
        opt.outputdir = opt.dataset + "_" + opt.mode
    opt.outputdir = get_dir(opt.outputdir)

    if opt.xp_time:
        opt.xp = opt.xp + "_" + get_time()
    if opt.xp_auto:
        opt.xp = get_time()
    if opt.auto_all:
        opt.outputdir = opt.dataset + "_" + opt.mode 
        opt.xp = get_time()
    opt.mode = opt.mode if opt.mode in ('refine', 'discover') else None
    opt.xp = 'revised-' + opt.xp
    opt.start = time_dir()
    start_st = datetime.datetime.now()
    opt.st = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
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

    setup, (train_data, test_data, validation_data), relations = get_stnn_data(opt.datadir, opt.dataset, opt.nt_train, opt.khop, opt.start_time, rescaled_method=opt.rescaled, normalize_method=opt.normalize_method, normalize=opt.normalize)
    # relations = relations[:, :, :, 0]
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    relations = relations.to(device)
    for k, v in setup.items():
        opt[k] = v

    # -- train inputs
    t_idx = torch.arange(opt.nt_train, out=torch.LongTensor()).unsqueeze(1).expand(opt.nt_train, opt.nx).contiguous()
    x_idx = torch.arange(opt.nx, out=torch.LongTensor()).expand_as(t_idx).contiguous()
    # dynamic
    idx_dyn = torch.stack((t_idx[1:], x_idx[1:])).view(2, -1).to(device)
    nex_dyn = idx_dyn.size(1)
    # decoder
    idx_dec = torch.stack((t_idx, x_idx)).view(2, -1).to(device)
    nex_dec = idx_dec.size(1)

    #######################################################################################################################
    # Model
    #######################################################################################################################
    model = SaptioTemporalNN_tanh(relations, opt.nx, opt.nt_train, opt.nd, opt.nz, opt.mode, opt.nhid, opt.nlayers, opt.dropout_f, opt.dropout_d, opt.activation, opt.periode).to(device)

    #######################################################################################################################
    # Optimizer
    #######################################################################################################################
    params = [{'params': model.factors_parameters(), 'weight_decay': opt.wd_z},
            {'params': model.dynamic.parameters()},
            {'params': model.decoder.parameters()}]
    if opt.mode in ('refine', 'discover'):
        params.append({'params': model.rel_parameters(), 'weight_decay': 0.})
        
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)
    elif opt.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=opt.lr, weight_decay=opt.wd)
    elif opt.optimizer == 'Rmsprop':
        optimizer = optim.RMSprop(params, lr=opt.lr, weight_decay=opt.wd)
    elif opt.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(params, lr=opt.lr, weight_decay=opt.wd)

    if opt.patience > 0:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience)


    #######################################################################################################################
    # Logs
    #######################################################################################################################
    logger = Logger(opt.outputdir, opt.xp, opt.checkpoint_interval)
    # with open(os.path.join(opt.outputdir, opt.xp, 'config.json'), 'w') as f:
    #     json.dump(opt, f, sort_keys=True, indent=4)


    #######################################################################################################################
    # Training
    #######################################################################################################################
    lr = opt.lr
    opt.mintest = 1000.0
    dyn_range = (opt.nt_train - 1) * opt.nx
    if command:
        pb = trange(opt.nepoch)
    else:
        pb = range(opt.nepoch)
    for e in pb:
        # ------------------------ Train ------------------------
        model.train()
        # --- decoder ---
        idx_perm = torch.randperm(nex_dec).to(device)
        batches = idx_perm.split(opt.batch_size)
        logs_train = defaultdict(float)
        for i, batch in enumerate(batches):
            optimizer.zero_grad()
            # data
            input_t = idx_dec[0][batch]
            input_x = idx_dec[1][batch]
            x_target = train_data[input_t, input_x]
            # closure
            x_rec = model.dec_closure(input_t, input_x)
            mse_dec = F.mse_loss(x_rec, x_target)
            # backward
            # mse_dec.backward()
            # step
            # logger.log('train_iter.mse_dec', mse_dec.item())
            logs_train['mse_dec'] += mse_dec.item() * len(batch)

            # dynamic
            # drop the idex outof dynamic range
            batch_dyn = [idx for idx in batch if idx < dyn_range]
            input_t = idx_dyn[0][batch_dyn]
            input_x = idx_dyn[1][batch_dyn]

            # closure
            z_inf = model.factors[input_t, input_x]
            z_pred = model.dyn_closure(input_t - 1, input_x)
            # loss
            mse_dyn = z_pred.sub(z_inf).pow(2).mean()
            loss_dyn = mse_dyn * opt.lambd
            if opt.l2_z > 0:
                loss_dyn += opt.l2_z * model.factors[input_t - 1, input_x].sub(model.factors[input_t, input_x]).pow(2).mean()
            if opt.mode in('refine', 'discover') and opt.l1_rel > 0:
                # rel_weights_tmp = model.rel_weights.data.clone()
                loss_dyn += opt.l1_rel * model.get_relations().abs().mean()
            loss_train = mse_dec + loss_dyn
            # backward
            loss_train.backward()
            # step
            optimizer.step()
            # clip
            # if opt.mode == 'discover' and opt.l1_rel > 0:  # clip
            #     sign_changed = rel_weights_tmp.sign().ne(model.rel_weights.data.sign())
            #     model.rel_weights.data.masked_fill_(sign_changed, 0)
            # log
            # logger.log('train_iter.mse_dyn', mse_dyn.item())
            logs_train['mse_dyn'] += mse_dyn.item() * len(batch)
            logs_train['loss_dyn'] += loss_dyn.item() * len(batch)

        # --- logs ---
        # TODO:
        logs_train['mse_dec'] /= nex_dec
        logs_train['mse_dyn'] /= nex_dyn
        logs_train['loss_dyn'] /= nex_dyn
        logs_train['loss'] = logs_train['mse_dec'] + logs_train['loss_dyn']
        logger.log('train_epoch', logs_train)
        # checkpoint
        # logger.log('train_epoch.lr', lr)
        logger.checkpoint(model)
        # ------------------------ Test ------------------------
        if opt.test:
            model.eval()
            with torch.no_grad():
                x_pred, _ = model.generate(opt.validation_length)
                score = rmse(x_pred, validation_data)
            if command:
                pb.set_postfix(loss=loss_train.item(), test=score)
            else:
                print(e, 'loss=', logs_train['loss'], 'test=', score)
            logger.log('test_epoch.rmse', score)
            if opt.mintest > score:
                opt.mintest = score
                # schedule lr
            if opt.patience > 0 and score < opt.sch_bound:
                lr_scheduler.step(score)
            lr = optimizer.param_groups[0]['lr']
            if lr <= 1e-5:
                break
        else:
            if command:
                pb.set_postfix(loss=loss_train.item())
            else:
                print(e, 'loss=', logs_train['loss'])
    # ------------------------ Test ------------------------
    model.eval()
    with torch.no_grad():
        x_pred, _ = model.generate(opt.nt - opt.nt_train)
        score_ts = rmse(x_pred, test_data, reduce=False)
        score = rmse(x_pred, test_data)
    # logger.log('test.rmse', score)
    # logger.log('test.ts', {t: {'rmse': scr.item()} for t, scr in enumerate(score_ts)})

    if opt.normalize == 'variance':
        true_pred_data = x_pred * opt.std + opt.mean
        true_score = score * opt.std
    elif opt.normalize_method == 'min_max':
        true_pred_data = x_pred * (opt.max - opt.min) + opt.mean
        true_score = score * (opt.max - opt.min)
    # save pred data
    for i in range(opt.nd):
        d_pred =true_pred_data[:,:, i].cpu().numpy()
        # print(d_pred)
        np.savetxt(os.path.join(get_dir(opt.outputdir), opt.xp, 'true_pred_' + str(i).zfill(3) +  '.txt'), d_pred, delimiter=',')

    opt.test_loss = score
    opt.true_loss = true_score
    logs_train['loss'] = logs_train['mse_dec'] + logs_train['loss_dyn']
    opt.train_loss = logs_train['loss']
    opt.end = time_dir()
    end_st = datetime.datetime.now()
    opt.et = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    opt.time = str(end_st - start_st)
    with open(os.path.join(get_dir(opt.outputdir), opt.xp, 'config.json'), 'w') as f:
        json.dump(opt, f, sort_keys=True, indent=4)
    logger.save(model)

if __name__ == "__main__":
    train(True)
