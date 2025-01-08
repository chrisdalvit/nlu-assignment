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

lr=0.00005
dropout=0.2
epochs=10
bert=bert-large-uncased

module load cuda/12.1

source ../../venv/bin/activate
python3.10 main.py --name base-dropout --lr $lr --dropout $dropout --num-epochs $epochs --bert-version $bert --save
			
deactivate
