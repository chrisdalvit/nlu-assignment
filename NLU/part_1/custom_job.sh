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

lr=0.0005
dropout=0.2
tbs=64
layers=2
optim=adam # one of [ sgd, adam ]

module load cuda/12.1

source ../../venv/bin/activate
python3.10 main.py --name bidirectional --bidirectional --optim $optim --lr $lr --num-layers $layers --train-batch-size $tbs --emb-dropout $dropout --hid-dropout $dropout --out-dropout $dropout --hid-size 200 --emb-size 300 --save

deactivate
