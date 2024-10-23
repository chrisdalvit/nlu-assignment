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

model=rnn
optim=sgd
out_dropout=0.0
emb_dropout=0.0
optim=sgd
lr=1.0

module load cuda/12.1

source ../../venv/bin/activate
python3.10 main.py --model $model --lr $lr --out-dropout $out_dropout --emb-dropout $emb_dropout --optim $optim
deactivate
