import argparse
import copy
from math import ceil

import torch
import torch.optim as optim
import numpy as np

from model import LM_RNN
from utils import Environment, Logger, NTAvgSGD
from functions import train_loop, eval_loop

parser = argparse.ArgumentParser()
parser.add_argument("--optim", default="sgd")
# Experiment also with a smaller or bigger model by changing hid and emb sizes
# A large model tends to overfit
parser.add_argument("--emb-size", type=int, default=300)
# With SGD try with an higher learning rate (> 1 for instance)
parser.add_argument("--lr", type=float, default=1.0)
# Clip the gradient
parser.add_argument("--clip", type=int, default=5)
# Don't forget to experiment with a lower training batch size
# Increasing the back propagation steps can be seen as a regularization step
parser.add_argument("--train-batch-size", type=int, default=64)
parser.add_argument("--dev-batch-size", type=int, default=128)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--out-dropout", type=float, default=0) 
parser.add_argument("--emb-dropout", type=float, default=0)

def run_epochs(model, optimizer, criterion_train, criterion_eval, env, logger, patience=3):
    best_ppl = float('inf')
    best_model = None
    current_patience = patience
    for epoch in range(env.args.epochs):
        loss_train = train_loop(env.dataloaders["train"], optimizer, criterion_train, model, env.args.clip)
        print(loss_train)
        ppl_dev, loss_dev = eval_loop(env.dataloaders["dev"], criterion_eval, model)
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            current_patience = patience
        else:
            current_patience -= 1
        logger.add_epoch_log(epoch, np.asarray(loss_train).mean(), np.asarray(loss_dev).mean(), ppl_dev)
        if current_patience <= 0:
            break
        
    best_model.to(env.device)
    return best_model

def main():
    args = parser.parse_args()
    env = Environment(args)
    logger = Logger(env)
    model = LM_RNN(
            "lstm",
            env.args.emb_size, 
            len(env.lang),
            weight_tying=True,
            variational_dropout=True
        ).to(env.device)
    
    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=env.pad_token_id)
    criterion_eval = torch.nn.CrossEntropyLoss(ignore_index=env.pad_token_id, reduction='sum')
    optimizer = None
    if env.args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=env.args.lr)
    elif env.args.optim == "nt-avgsgd":
        optimizer = NTAvgSGD(
            model, 
            env.dataloaders["dev"],
            criterion_eval,
            env.args.lr,
            L = ceil(len(env.dataloaders["train"]) / env.args.train_batch_size),
            n = 5
        )
        
    best_model = run_epochs(model, optimizer, criterion_train, criterion_eval, env, logger)
    final_ppl, _ = eval_loop(env.dataloaders["test"], criterion_eval, best_model)
    logger.set_final_ppl(final_ppl)
    print(logger.dumps())
    
if __name__ == "__main__":
    main()