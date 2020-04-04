import numpy as np
import torch
from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, input_size, hid_size, hid_layers, output_size, seq_length):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(input_size, hid_size, hid_layers, dropout=0.5, batch_first = True)
        self.linear = nn.Linear(hid_size, output_size)
        self.out = nn.Linear(seq_length, 1)
        self.input_size = input_size
        self.hid_size = hid_size
        self.hid_layers = hid_layers
        self.seq_length = seq_length
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out.weight)
        self.out.bias.data.fill_(0)
        nn.init.xavier_normal_(self.rnn.all_weights[0][0])
        nn.init.xavier_normal_(self.rnn.all_weights[0][1])
        nn.init.xavier_normal_(self.rnn.all_weights[1][0])
        nn.init.xavier_normal_(self.rnn.all_weights[1][1])


    def forward(self, input):
        nd = input.size(-1)
        nx = input.size(-2)
        input = input.contiguous()
        input = input.view(-1, self.seq_length, nd*nx)
        out, _ = self.rnn(input)
        outs = []
        for timestep in range(input.size(1)):
            outs.append(torch.relu(self.linear(out[:,timestep,:])))
        new_out = torch.stack(outs, dim=2)
        outs = []
        for batch in range(input.size(0)):
            outs.append(torch.relu(self.out(new_out[batch])).view(-1,1))
        output = torch.stack(outs, dim=0).contiguous()
        output = output.view(-1, nx, nd)
        return output
        
    def update(self, init, pred):
        # init : seq_len , nx, nd
        nd = init.size(-1)
        nx = init.size(-2)
        new_seq = []
        for i in range(self.seq_length-1):
            new_seq.append(init[i+1])
        new_seq.append(pred.squeeze(0))
        # print('pred:', pred.size())
        # print('init:', init[0].size())
        return torch.stack(new_seq, dim=0) # seq_len, nx, nd
    
    def generate(self, init, length):
        nd = init.size(-1)
        nx = init.size(-2)
        pred_list = []
        for i in range(length):
            new_pred = self.forward(init.unsqueeze(0)) # 1, nx, nd
            pred_list.append(new_pred)
            init = self.update(init, new_pred)
        return torch.cat(pred_list, dim=0)


class LSTMNet_onelinear(nn.Module):
    def __init__(self, input_size, hid_size, hid_layers, output_size, seq_length):
        super(LSTMNet_onelinear, self).__init__()
        self.rnn = nn.LSTM(input_size, hid_size, hid_layers, dropout=0.5, batch_first = True)
        self.linear = nn.Linear(hid_size * seq_length, output_size)
        self.input_size = input_size
        self.hid_size = hid_size
        self.hid_layers = hid_layers
        self.seq_length = seq_length
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        nn.init.xavier_normal_(self.rnn.all_weights[0][0])
        nn.init.xavier_normal_(self.rnn.all_weights[0][1])
        nn.init.xavier_normal_(self.rnn.all_weights[1][0])
        nn.init.xavier_normal_(self.rnn.all_weights[1][1])


    def forward(self, input):
        nd = input.size(-1)
        nx = input.size(-2)
        input = input.contiguous()
        input = input.view(-1, self.seq_length, nd*nx)
        out, _ = self.rnn(input)
        out = out.contiguous()
        out = out.view(-1, self.seq_length * self.hid_size)
        out = out.contiguous()
        out = self.linear(out).view(-1, nx, nd)
        return torch.relu(out)
        
    def update(self, init, pred):
        # init : seq_len , nx, nd
        nd = init.size(-1)
        nx = init.size(-2)
        new_seq = []
        for i in range(self.seq_length-1):
            new_seq.append(init[i+1])
        new_seq.append(pred.squeeze(0))
        # print('pred:', pred.size())
        # print('init:', init[0].size())
        return torch.stack(new_seq, dim=0) # seq_len, nx, nd
    
    def generate(self, init, length):
        nd = init.size(-1)
        nx = init.size(-2)
        pred_list = []
        for i in range(length):
            new_pred = self.forward(init.unsqueeze(0)) # 1, nx, nd
            pred_list.append(new_pred)
            init = self.update(init, new_pred)
        return torch.cat(pred_list, dim=0)

class GRUNet(nn.Module):
    def __init__(self, input_size, hid_size, hid_layers, output_size, seq_length):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(input_size, hid_size, hid_layers, dropout=0.5, batch_first = True)
        self.linear = nn.Linear(hid_size, output_size)
        self.out = nn.Linear(seq_length, 1)
        self.input_size = input_size
        self.hid_size = hid_size
        self.hid_layers = hid_layers
        self.seq_length = seq_length
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out.weight)
        self.out.bias.data.fill_(0)
        nn.init.xavier_normal_(self.rnn.all_weights[0][0])
        nn.init.xavier_normal_(self.rnn.all_weights[0][1])
        nn.init.xavier_normal_(self.rnn.all_weights[1][0])
        nn.init.xavier_normal_(self.rnn.all_weights[1][1])


    def forward(self, input):
        nd = input.size(-1)
        nx = input.size(-2)
        input = input.contiguous()
        input = input.view(-1, self.seq_length, nd*nx)
        out, _ = self.rnn(input)
        outs = []
        for timestep in range(input.size(1)):
            outs.append(torch.relu(self.linear(out[:,timestep,:])))
        new_out = torch.stack(outs, dim=2)
        outs = []
        for batch in range(input.size(0)):
            outs.append(torch.relu(self.out(new_out[batch])).view(-1,1))
        output = torch.stack(outs, dim=0).contiguous()
        output = output.view(-1, nx, nd)
        return output
        
    def update(self, init, pred):
        # init : seq_len , nx, nd
        nd = init.size(-1)
        nx = init.size(-2)
        new_seq = []
        for i in range(self.seq_length-1):
            new_seq.append(init[i+1])
        new_seq.append(pred.squeeze(0))
        # print('pred:', pred.size())
        # print('init:', init[0].size())
        return torch.stack(new_seq, dim=0) # seq_len, nx, nd
    
    def generate(self, init, length):
        nd = init.size(-1)
        nx = init.size(-2)
        pred_list = []
        for i in range(length):
            new_pred = self.forward(init.unsqueeze(0)) # 1, nx, nd
            pred_list.append(new_pred)
            init = self.update(init, new_pred)
        return torch.cat(pred_list, dim=0)

class GRUNet_onelinear(nn.Module):
    def __init__(self, input_size, hid_size, hid_layers, output_size, seq_length):
        super(GRUNet_onelinear, self).__init__()
        self.rnn = nn.GRU(input_size, hid_size, hid_layers, dropout=0.5, batch_first = True)
        self.linear = nn.Linear(hid_size * seq_length, output_size)
        self.input_size = input_size
        self.hid_size = hid_size
        self.hid_layers = hid_layers
        self.seq_length = seq_length
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        nn.init.xavier_normal_(self.rnn.all_weights[0][0])
        nn.init.xavier_normal_(self.rnn.all_weights[0][1])
        nn.init.xavier_normal_(self.rnn.all_weights[1][0])
        nn.init.xavier_normal_(self.rnn.all_weights[1][1])


    def forward(self, input):
        nd = input.size(-1)
        nx = input.size(-2)
        input = input.contiguous()
        input = input.view(-1, self.seq_length, nd*nx)
        out, _ = self.rnn(input)
        out = out.contiguous()
        out = out.view(-1, self.seq_length * self.hid_size)
        out = out.contiguous()
        out = self.linear(out).view(-1, nx, nd)
        return torch.relu(out)
        
    def update(self, init, pred):
        # init : seq_len , nx, nd
        nd = init.size(-1)
        nx = init.size(-2)
        new_seq = []
        for i in range(self.seq_length-1):
            new_seq.append(init[i+1])
        new_seq.append(pred.squeeze(0))
        # print('pred:', pred.size())
        # print('init:', init[0].size())
        return torch.stack(new_seq, dim=0) # seq_len, nx, nd
    
    def generate(self, init, length):
        nd = init.size(-1)
        nx = init.size(-2)
        pred_list = []
        for i in range(length):
            new_pred = self.forward(init.unsqueeze(0)) # 1, nx, nd
            pred_list.append(new_pred)
            init = self.update(init, new_pred)
        return torch.cat(pred_list, dim=0)