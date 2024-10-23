#!/bin/bash

#SBATCH --job-name=nlu1
#SBATCH --output=output/custom.json
#SBATCH --error=output/custom.err
#SBATCH --partition=edu-20h
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --account=giuseppe.riccardi.edu

optim=sgd
out_dropout=0.0
emb_dropout=0.0
hid_dropout=0.0
num_layers=2
epochs=100
emb_size=400
lr=10.0

module load cuda/12.1

source ../../venv/bin/activate
python3.10 main.py job.sh --optim $optim \
			--weight-tying \
			--variational-dropout \
			--emb-dropout $emb_dropout \
			--out-dropout $out_dropout \
			--hid-dropout $hid_dropout \
			--num-layers $num_layers \
			--epochs $epochs \
			--emb-size $emb_size \
			--train-batch-size $train_batch_size \
			--lr $lr
deactivate
