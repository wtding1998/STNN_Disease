import numpy
import torch
import matplotlib.pyplot as plt


from get_dataset import get_time_data

class SEIR(torch.nn.Module):
    def __init__(self, N0, E0, I0, R0, t):
        super(SEIR, self).__init__()
        self.t = t # 1天离散的时间间隔数
        self.delta = 1.0 / t
        # 初始参数
        self.N0 = torch.tensor(N0).float()  # 人口总数
        self.E0 = torch.tensor(E0).float()  # 潜伏者
        self.I0 = torch.tensor(I0).float() # 感染者
        self.R0 = torch.tensor(R0).float()  # 康复者
        self.S0 = self.N0 - self.I0 - self.E0 - self.R0  # 易感者

        # 待定
        self.r = torch.nn.Parameter(torch.tensor(1.0))  #感染者接触易感者的人数
        self.B = torch.nn.Parameter(torch.tensor(0.02))    #传染概率
        self.a = torch.nn.Parameter(torch.tensor(0.01))     #潜伏者转化为感染者概率
        self.r2 = torch.nn.Parameter(torch.tensor(2.0))     #潜伏者接触易感者的人数
        self.B2 = torch.nn.Parameter(torch.tensor(0.03))   #潜伏者传染正常人的概率
        self.y = torch.nn.Parameter(torch.tensor(0.1))     #康复概率
    
    def forward(self, S, E, I, R):
        # N = S + E + I + R
        for i in range(t):
            new_S = S - self.delta * self.r * self.B * S * I / self.N0 - self.delta * self.r2 * self.B2 * S * E / self.N0
            new_E = E + self.delta * self.r * self.B * S * I / self.N0 - self.delta * self.a * E + self.delta * self.r2 * self.B2 * S * E / self.N0
            new_I = I + self.delta * self.a * E - self.delta * self.y * I 
            new_R = R + self.delta * self.y * I
            S = new_S
            E = new_E
            I = new_I
            R = new_R
        return S, E, I, R

    def pred(self, days):
        # 改成列表
        S = [self.S0]
        E = [self.E0]
        I = [self.I0]
        R = [self.R0]
        for day in range(days):
            # print(model(S[-1], E[-1], I[-1], R[-1]))
            new_S, new_E, new_I, new_R = self.forward(S[-1], E[-1], I[-1], R[-1])
            S.append(new_S)
            E.append(new_E)
            I.append(new_I)
            R.append(new_R)
        S = torch.stack(S, dim=0)
        E = torch.stack(E, dim=0)
        I = torch.stack(I, dim=0)
        R = torch.stack(R, dim=0)
        return S, E, I, R

def plot(S, E, I, R, days):
    T = numpy.arange(days + 1)
    figure = plt.figure()
    plt.plot(T, S.detach().numpy(), label='Susceptible')
    plt.plot(T, E.detach().numpy(), label='Exposed')
    plt.plot(T, I.detach().numpy(), label='infectious')
    plt.plot(T, R.detach().numpy(), label='Recovered')
    plt.title("SEIR sample")
    xl = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xl)
    plt.legend()
    plt.show()
    
 
if __name__ == "__main__":
    # === sample ===
    # t = 200
    # days = 100
    # N0 = 10000
    # E0 = 0
    # I0 = 1
    # R0 = 0
    # model = SEIR(N0, E0, I0, R0, t)
    # S, E, I, R = model.pred(days)
    # plot(S, E, I, R, days)

    # === data ===
    nepoch = 1000
    t = 10
    data = get_time_data('data', 'ncov_sum')
    confirmed_data = data[:, 0, 0]
    cured_data = data[:, 0, 1]
    days = confirmed_data.size(0) - 1
    N0 = 1.6e9
    E0 = 0
    I0 = confirmed_data[0]
    R0 = cured_data[0]
    # === model ===
    model = SEIR(N0, E0, I0, R0, t)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss = torch.nn.MSELoss()

    # for name, param in model.named_parameters():
    #     print(name, param)
    # === train ===
    model.train()
    try:
        for i in range(nepoch):
            optim.zero_grad()
            S, E, I, R = model.pred(days)
            # train_loss1 = loss(I[:10], confirmed_data[:10]) + loss(R[:10], cured_data[:10])
            # train_loss2 = loss(I[10:], confirmed_data[10:]) + loss(R[10:], cured_data[10:])
            # train_loss = train_loss1 + 10.0 * train_loss2
            train_loss = loss(I, confirmed_data) + loss(R, cured_data)
            train_loss.backward()
            # 这里norm_type可以选择L1范数，L2范数和无穷范数，分别对应`1, 2, 'inf'`
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
            optim.step()
            print(i, train_loss.item())

    except:
        for name, param in model.named_parameters():
            print(name, param)
        plot(I, E, I, R, days)
        S, E, I, R = model.pred(days + 1)
        print(I)    
        print(R)  

    for name, param in model.named_parameters():
        print(name, param)
    plot(I, E, I, R, days)
    S, E, I, R = model.pred(days + 1)
    print(I)    
    print(R)    