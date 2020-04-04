import matplotlib.pyplot as plt
import numpy as np
import time
from utils import DotDict
from get_dataset import get_time_data
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.optimizers import SGD, RMSprop, adam

np.random.seed(2017)

def split_data(dataset='ncov',sequence_length=2,nt_train=55):

    raw_data = np.array(get_time_data('data', dataset)) 
    opt = DotDict()
    opt.nt = raw_data.shape[0]
    opt.nx = raw_data.shape[1] 
    opt.nd = raw_data.shape[2]
    raw_data = np.reshape(raw_data, (opt.nt, opt.nx * opt.nd))
    print ("Data loaded from csv. Formatting...")
    result = []
    for index in range(len(raw_data) - sequence_length + 1):
        result.append(raw_data[index: index + sequence_length])
    result = np.array(result)
    result_mean = result.mean()
    result -= result_mean
    result_std = result.std()
    result = result / result_std
    train = result[:nt_train]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[nt_train:, :-1] 
    y_test = result[nt_train:, -1]
    # print(X_test.shape)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], opt.nx*opt.nd))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], opt.nx*opt.nd))
    return opt, (X_train, y_train, X_test, y_test, result_mean)


def build_model(opt):
    model = Sequential()
    layers = [opt.nd*opt.nx, 50, 100, opt.nd*opt.nx]

    model.add(GRU(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    # optim = RMSprop(lr=0.001)
    model.compile(loss="mse", optimizer='rmsprop', metrics=['mae', 'mape'])
    print ("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, data=None, dataset='ncov_confirmed'):
    global_start_time = time.time()
    epochs = 10
    nt_train = 50
    sequence_length = 2
    # dataset = 'ncov_confirmed'

    if data is None:
        print ('Loading data... ')
        opt, (X_train, y_train, X_test, y_test, result_mean) = split_data(
            dataset, sequence_length, nt_train)
    else:
        X_train, y_train, X_test, y_test = data

    print ('\nData Loaded. Compiling...\n')
    print(X_train.shape) # (50, 1, 31)
    print(y_train.shape) # (50, 31)
    # model = None
    if model is None:
        model = build_model(opt)
    try:
        model.fit(
            X_train, y_train,
            batch_size=8, epochs=epochs, validation_split=0.05)
        # ! save model
        model.save('keras_model.h5')
        predicted = model.predict(X_test)  
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        # Evaluate
        scores = model.evaluate(X_test, y_test, batch_size=8)
        print("\nevaluate result: \nmse={:.6f}\nmae={:.6f}\nmape={:.6f}".format(scores[0], scores[1], scores[2]))

        # draw the figure
        y_test += result_mean
        predicted += result_mean
        nt_test = y_test.shape[0]
        y_test = np.reshape(y_test, (nt_test, opt.nx, opt.nd))
        predicted = np.reshape(predicted, (nt_test, opt.nx, opt.nd))
        print ('Training duration (s) : ', time.time() - global_start_time)
        show_plt(y_test, predicted)
        predicted = np.reshape(predicted, (opt.nx, opt.nd))
        np.savetxt('pred.txt', predicted, delimiter=',')
        # "confirm, cure, dead"
        confirmed_pred = predicted[:, 0]
        cure_pred = predicted[:, 1]
        dead_pred = predicted[:, 2]
        print("pred : \nconfirmed={:.1f}\ncured={:.1f}\ndead={:.1f}".format(confirmed_pred.sum(), cure_pred.sum(), dead_pred.sum()))
    except Exception as e:
        print ('Training duration (s) : ', time.time() - global_start_time)
        print (str(e))

    return model, y_test, predicted


def show_plt(y_test, predict):
    # (nt_test, nx, nd)
    nt_test, nx, nd = y_test.shape
    # print(y_test.shape)
    for i in range(nd):
        # print(0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(list(range(nx)), y_test[0,:,i],label="Real")
        ax.legend(loc='upper left')
        plt.scatter(list(range(nx)), predict[0,:,i],label="Prediction")
        plt.legend(loc='upper left')
        plt.show()


if __name__ == '__main__':
    run_network(dataset='ncov_confirmed')
