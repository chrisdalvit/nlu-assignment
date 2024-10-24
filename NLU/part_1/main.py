import copy
import argparse
from functools import partial

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from functions import train_loop, eval_loop, init_weights
from utils import IntentsAndSlots, Environment, Logger, collate_fn
from model import ModelIAS


parser = argparse.ArgumentParser()
parser.add_argument("--hid-size", type=int, default=200)
parser.add_argument("--emb-size", type=int, default=300)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--clip", type=int, default=5)
parser.add_argument("--num-layers", type=int, default=1)
parser.add_argument("--train-batch-size", type=int, default=128)
parser.add_argument("--emb-dropout", type=float, default=0.0)
parser.add_argument("--out-dropout", type=float, default=0.0)
parser.add_argument("--hid-dropout", type=float, default=0.0)
parser.add_argument("--bidirectional", action='store_true')
parser.add_argument("--vdropout", action='store_true')

def main():
    args = parser.parse_args()
    env = Environment(args)
    logger = Logger(env)
    lang = env.lang

    model = ModelIAS(
        env.args.hid_size, 
        len(lang.slot2id), 
        len(lang.intent2id), 
        env.args.emb_size, 
        len(lang.word2id), 
        env.pad_token,
        bidirectional=env.args.bidirectional,
        emb_dropout=env.args.emb_dropout,
        out_dropout=env.args.out_dropout,
        hid_dropout=env.args.hid_dropout,
        n_layer=env.args.num_layers,
        variational_dropout=env.args.vdropout
    ).to(env.device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=env.args.lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=env.pad_token)
    criterion_intents = nn.CrossEntropyLoss()
    
    train_dataset = IntentsAndSlots(env.train_raw, lang)
    dev_dataset = IntentsAndSlots(env.dev_raw, lang)
    test_dataset = IntentsAndSlots(env.test_raw, lang)
    train_loader = DataLoader(train_dataset, batch_size=env.args.train_batch_size, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device))
    
    n_epochs = 200
    patience = 3
    best_model = None
    
    best_f1 = 0
    for epoch in range(1,n_epochs):
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=env.args.clip)
        if epoch % 5 == 0: # We check the performance every 5 epochs
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            
            f1 = results_dev['total']['f']
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0:
                break
            logger.add_epoch_log(epoch, np.asarray(loss).mean(), np.asarray(loss_dev).mean(), f1) 
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model.to(env.device), lang)    
    logger.set_final_scores(results_test['total']['f'], intent_test['accuracy'])
    print(logger.dumps())
    
if __name__ == "__main__":
    main()
    