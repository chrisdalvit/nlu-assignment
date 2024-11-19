import copy
import argparse
from functools import partial

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from functions import train_loop, eval_loop
from utils import Logger, Environment, IntentsAndSlots, collate_fn
from model import ModelIAS

BERT_VERSIONS = [
    "bert-base-uncased", 
    "bert-large-uncased", 
    "bert-tiny-uncased", 
    "bert-small-uncased", 
    "bert-medium-uncased", 
    "bert-mini-uncased"
]

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--num-runs", type=str, default=1)
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--train-batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--bert-version", type=str, choices=BERT_VERSIONS)


def run_epochs(run, model, train_loader, dev_loader, optimizer, criterion_slots, criterion_intents, env, logger):
    """Run all epochs.

    Args:
        run: Index of current run.
        model: PyTorch model.
        train_loader: Train dataloader.
        dev_loader: Evaluation dataloader.
        optimizer: Optimizer.
        criterion_slots: Loss funciton for slot filling.
        criterion_intents: Loss function for intent classification.
        env: Training environment.
        logger: Logger for metric logging.

    Returns:
        PyTorch model: Best model.
    """
    for epoch in range(env.args.num_epochs):
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, env)
        results_dev, _, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, env)
        f1 = results_dev['total']['f']
        best_model = copy.deepcopy(model)
        logger.add_epoch_log(run, epoch, np.asarray(loss).mean(), np.asarray(loss_dev).mean(), f1)
    return best_model


def main():
    args = parser.parse_args()
    env = Environment(args)
    logger = Logger(env)
    out_int = len(env.lang.intent2id)
    out_slot = len(env.lang.slot2id)
    
    for run in range(env.args.num_runs):
        logger.add_run()
        model = ModelIAS(out_slot, out_int, version=env.args.bert_version, dropout=env.args.dropout)
        
        criterion_slots = nn.CrossEntropyLoss(ignore_index=env.pad_token)
        criterion_intents = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=env.args.lr)
        
        train_dataset = IntentsAndSlots(env.train_raw, env.lang)
        dev_dataset = IntentsAndSlots(env.dev_raw, env.lang)
        test_dataset = IntentsAndSlots(env.test_raw, env.lang)
        train_loader = DataLoader(train_dataset, batch_size=env.args.train_batch_size, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device),  shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device))
        test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device))
        
        best_model = run_epochs(run, model, train_loader, dev_loader, optimizer, criterion_slots, criterion_intents, env, logger)
        if best_model:
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, env)
            logger.set_final_scores(run, results_test['total']['f'], intent_test['accuracy'])
        else:
            logger.set_final_scores(run, 0, 0)
    print(logger.dumps())
        
if __name__ == "__main__":
    main()
