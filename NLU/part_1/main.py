import argparse
from functools import partial

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from functions import train_loop, eval_loop, init_weights
from utils import IntentsAndSlots, Environment, collate_fn
from model import ModelIAS


parser = argparse.ArgumentParser()
parser.add_argument("--hid-size", type=int, default=200)
parser.add_argument("--emb-size", type=int, default=300)
parser.add_argument("--lr", type=int, default=0.0001)
parser.add_argument("--clip", type=int, default=5)

def main():
    args = parser.parse_args()
    env = Environment(args)
    lang = env.lang

    model = ModelIAS(
        env.args.hid_size, 
        len(lang.slot2id), 
        len(lang.intent2id), 
        env.args.emb_size, 
        len(lang.word2id), 
        env.pad_token
    ).to(env.device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=env.args.lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=env.pad_token)
    criterion_intents = nn.CrossEntropyLoss()
    
    train_dataset = IntentsAndSlots(env.train_raw, lang)
    dev_dataset = IntentsAndSlots(env.dev_raw, lang)
    test_dataset = IntentsAndSlots(env.test_raw, lang)
    
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device))
    
    n_epochs = 200
    patience = 3
    losses_train, losses_dev, sampled_epochs = [], [], []
    best_f1 = 0
    for x in range(1,n_epochs):
        print(f"Epoch {x}...")
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=env.args.clip)
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0:
                break

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])
        
if __name__ == "__main__":
    main()
    