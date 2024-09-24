#!/bin/bash

#SBATCH --job-name=nlu1

#SBATCH --partition=edu-20h
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --account=giuseppe.riccardi.edu

module load cuda/12.1

source ../../venv/bin/activate
python3.10 main.py --model $1 --lr $2 --out-dropout $3 --emb-dropout $3 --optim $4
deactivate
