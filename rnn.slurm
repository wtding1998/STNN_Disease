#!/bin/bash

#SBATCH --job-name=dwt_rnn
#SBATCH --partition=small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=end
#SBATCH --mail-user=569018166@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --err=%j.err
#SBATCH --time=00:30:00

python train_rnn.py --nepoch 120000 --rnn_model LSTM --outputdir output_aids --xp test --device -1
