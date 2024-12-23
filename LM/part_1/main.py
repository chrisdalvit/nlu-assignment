import argparse
import copy

import numpy as np
import torch
import torch.optim as optim

from utils import Environment, Logger
from functions import train_loop, eval_loop, init_weights
from model import LM_RNN

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="rnn")
parser.add_argument("--optim", default="sgd")
parser.add_argument("--hid-size", type=int, default=200)
parser.add_argument("--emb-size", type=int, default=300)
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--clip", type=int, default=5)
parser.add_argument("--train-batch-size", type=int, default=64)
parser.add_argument("--dev-batch-size", type=int, default=128)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--out-dropout", type=float, default=0) 
parser.add_argument("--emb-dropout", type=float, default=0)


def run_epochs(model, optimizer, criterion_train, criterion_eval, env, logger, patience=3):
    """Run all epochs.

    Args:
        model: PyTorch model
        optimizer: Optimizer for weight update
        criterion_train: Training loss function
        criterion_eval: Evaluation loss function
        env: Training environment information
        logger: Logger
        patience (int, optional): Maximum patience count. Defaults to 3.

    Returns:
        nn.Module: Best model
    """
    best_ppl = float('inf')
    best_model = None
    current_patience = patience
    for epoch in range(env.args.epochs):
        loss_train = train_loop(env.dataloaders["train"], optimizer, criterion_train, model, env.args.clip)
        ppl_dev, loss_dev = eval_loop(env.dataloaders["dev"], criterion_eval, model)
        if  ppl_dev < best_ppl:
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
            env.args.model,
            env.args.emb_size, 
            env.args.hid_size, 
            len(env.lang), 
            emb_dropout=env.args.emb_dropout,
            out_dropout=env.args.out_dropout,
            pad_index=env.pad_token_id
        ).to(env.device)
    model.apply(init_weights)
    optimizer = None
    if env.args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=env.args.lr)
    elif env.args.optim == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=env.args.lr)

    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=env.pad_token_id)
    criterion_eval = torch.nn.CrossEntropyLoss(ignore_index=env.pad_token_id, reduction='sum')
    best_model = run_epochs(model, optimizer, criterion_train, criterion_eval, env, logger)
    final_ppl, _ = eval_loop(env.dataloaders["test"], criterion_eval, best_model)
    logger.set_final_ppl(final_ppl)
    print(logger.dumps())

if __name__ == "__main__":
    main()