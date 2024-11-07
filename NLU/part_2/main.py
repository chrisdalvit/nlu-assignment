import copy
import argparse
from functools import partial

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from functions import train_loop, eval_loop
from utils import Environment, IntentsAndSlots, collate_fn
from model import ModelIAS

parser = argparse.ArgumentParser()

def main():
    lr = 0.0001
    args = parser.parse_args()
    env = Environment(args)
    out_int = len(env.lang.intent2id)
    out_slot = len(env.lang.slot2id)
    model = ModelIAS(out_slot, out_int)
    
    criterion_slots = nn.CrossEntropyLoss(ignore_index=env.pad_token)
    criterion_intents = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = IntentsAndSlots(env.train_raw, env.lang)
    dev_dataset = IntentsAndSlots(env.dev_raw, env.lang)
    test_dataset = IntentsAndSlots(env.test_raw, env.lang)
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=env.pad_token, device=env.device))
    
    n_epochs = 200
    patience = 15
    best_model = None
    for epoch in range(1, n_epochs):
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        results_dev, _, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, env.lang)
        f1 = results_dev['total']['f']
        if epoch > 5 and f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)
            patience = 3
        else:
            patience -= 1
        if patience <= 0:
            break
    if best_model:
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, env.lang)
        print(results_test['total']['f'], intent_test['accuracy'])    
        
if __name__ == "__main__":
    main()
