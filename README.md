# Predict the Distribution of Disease by RNN and STNN

This project is supported by the National Natural Science Foundation of China (Grant No: 11601327) and the Key Construction National “985” Program of China (Grant No: WF220426001).


## Data
### Aids
The file `aids.csv` contains the raw temperature data. The 156 rows correspond to the 156 timestep, and the 29 columns are the 29 space points.
The file `aids_relations.csv` contains the spatial relation between the 29 space points. It is a 29 by 29 adjacency matrix _A_, where _A(i, j)_ = 1 means that series _i_ is a direct neighbor of series _j_ in space, and is 0 otherwise.
### Flu
The file `flu.csv` contains the raw temperature data. The 156 rows correspond to the 156 timestep, and the 29 columns are the 29 space points.
The file `flu_relations.csv` contains the spatial relation between the 29 space points. It is a 29 by 29 adjacency matrix _A_, where _A(i, j)_ = 1 means that series _i_ is a direct neighbor of series _j_ in space, and is 0 otherwise.
### Heat
The file `heat.csv` contains the raw temperature data. The 200 rows correspond to the 200 timestep, and the 41 columns are the 41 space points.
The file `heat_relations.csv` contains the spatial relation between the 41 space points. It is a 41 by 41 adjacency matrix _A_, where _A(i, j)_ = 1 means that series _i_ is a direct neighbor of series _j_ in space, and is 0 otherwise.
## Model
### Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relation Discovery

ICDM 2018 - IEEE International Conference on Data Mining series (ICDM)

[Conference Paper](https://ieeexplore.ieee.org/document/8215543/)

[Journal Extension](https://link.springer.com/article/10.1007/s10115-018-1291-x)

Commands for reproducing synthetic experiments:

#### STNN
`python train_stnn.py --dataset aids --outputdir output_aids --manualSeed 1932 --xp stnn`

`python train_stnn.py --dataset flu --outputdir output_flu --manualSeed 7011 --xp stnn`

`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 2021 --xp stnn`

#### STNN-R(efine)
`python train_stnn.py --dataset aids --outputdir output_aids --manualSeed 3301 --xp stnn_r --mode refine --patience 800 --l1_rel 1e-8`

`python train_stnn.py --dataset flu --outputdir output_flu --manualSeed 3796 --xp stnn_r --mode refine --patience 800 --l1_rel 1e-8`

`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 5718 --xp stnn_r --mode refine --patience 800 --l1_rel 1e-8`
#### STNN-D(iscovery)
`python train_stnn.py --dataset aids --outputdir output_aids --manualSeed 1290 --xp stnn_d --mode discover --patience 1000 --l1_rel 3e-6`

`python train_stnn.py --dataset flu --outputdir output_flu --manualSeed 8837 --xp stnn_d --mode discover --patience 1000 --l1_rel 3e-6`

`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 9690 --xp stnn_d --mode discover --patience 1000 --l1_rel 3e-6`
<!-- ## Modulated Heat Diffusion
### STNN
`python train_stnn.py --dataset heat_m --outputdir output_heat_m --manualSeed 679 --xp stnn`

### STNN-R(efine)
`python train_stnn.py --dataset heat_m --outputdir output_heat_m --manualSeed 3488 --xp stnn_r --mode refine --l1_rel 1e-5`

### STNN-D(iscovery)
`python train_stnn_.py --dataset heat_m --outputdir output_m --xp test --manualSeed 7664 --mode discover --patience 500 --l1_rel 3e-6` -->

### RNN
Here **LSTM** and **GRU** are used.

Commands for reproducing synthetic experiments:
#### LSTM
`python train_rnn.py --dataset aids --model LSTM --manualSeed 1208 --xp LSTM_aids`

`python train_rnn.py --dataset flu --model LSTM --manualSeed 1471 --xp LSTM_flu`

`python train_rnn.py --dataset heat --model LSTM --manualSeed 6131 --xp LSTM_heat`
#### GRU
`python train_rnn.py --dataset aids --model GRU --manualSeed 1208 --xp GRU_aids`

`python train_rnn.py --dataset flu --model GRU --manualSeed 1471 --xp GRU_flu`

`python train_rnn.py --dataset heat --model GRU --manualSeed 6131 --xp GRU_heat`