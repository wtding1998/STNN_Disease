# Todo
## Code
- [x] add GRU model
    - [x] add GRU model in rnn_model.py
    - [x] add option in train_rnn.py
    - [x] modify model in train_rnn.py
- [x] add script
- [x] the name of  outputdir
- [x] auto name mode
- [x] make another dir for output
    - [x] os find the dir
- [x] update the git in server
- [ ] complete batch_size
- [ ] weight decay
- [ ] add normalize
- khop?

**result** : 
- [x] print the information for the given model | *test_time* 
- [x] summary the information of the total folder and get the best result | *nhid*
- [x] summary the information of the total mode | *aids_LSTM*
- [x] change the time for logs
- [x] add the mean to the beginning of the df
- [x] **write the information to be a class**
    - init with folder
    - info
- [x] save and load model
- [x] add pred for STNN
- [ ] load model for STNN
- [ ] enumerate exp by config
- [ ] RNN_F
- [ ] combine RNN and STNN
- [x] for the given model list, finish the jupyter notebook to get the information of them.
- [x] add start time and end time in config

something should left now : 
- [x] add test loss in config.json
- [x] add a file to get the best consequence
- [ ] draw the picture of test_loss
- [ ] complete the code for n-dim data

## To improve STNN:
STNN will weak in some epochs
1. maybe watch the parameters in every model may be 0
1. maybe add test set can improve the peformance

## data 
- [ ] the time in time

## report 
### experiments
- [ ] compare the short term performance with RNN
    - [ ] for aids, flu, heat nt_train 150
        - aids_rnn : 
        - aids_stnn :
- [ ] for smooth data
    - [x] heat with LSTM
    - [ ] heat with STNN
- [ ] dim of latent factor
    - test in heat dataset 
    - make a summary
- [ ] decrease the constraint in dynamics may add variance in time series
    - or the decoder is near 0-map
- [ ] dropout:
    - when dropout_d = dropout_f = 0.5 : the train loss will be about 230 all the time
    - when dropout_d = dropout_f = 0.3 : the train loss will be about 160 all the time
    - when dropout_d = dropout_f = 0.1 : the train loss will be about 50 all the time
    - when the nlayer and nhid larger, what will be ?
    - so what they should be?
