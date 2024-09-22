import os
import copy
from functools import partial
import math

import toml
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim

from utils.utils import train_loop, eval_loop
from utils.environment import Environment
from models import get_model

# Don't forget to experiment with a lower training batch size
# Increasing the back propagation steps can be seen as a regularization step
env = Environment()

# Experiment also with a smaller or bigger model by changing hid and emb sizes
# A large model tends to overfit
hid_size = 200
emb_size = 300

# With SGD try with an higher learning rate (> 1 for instance)
lr = 1.1 # This is definitely not good for SGD
clip = 5 # Clip the gradient

model = get_model("rnn", emb_size, hid_size, env)

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion_train = torch.nn.CrossEntropyLoss(ignore_index=env.pad_token_id)
criterion_eval = torch.nn.CrossEntropyLoss(ignore_index=env.pad_token_id, reduction='sum')

n_epochs = 100
patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_ppl = math.inf
best_model = None
pbar = tqdm(range(1,n_epochs))
#If the PPL is too high try to change the learning rate
for epoch in pbar:
    loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
    sampled_epochs.append(epoch)
    losses_train.append(np.asarray(loss).mean())
    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
    losses_dev.append(np.asarray(loss_dev).mean())
    pbar.set_description("PPL: %f" % ppl_dev)
    if  ppl_dev < best_ppl: # the lower, the better
        best_ppl = ppl_dev
        best_model = copy.deepcopy(model).to('cpu')
        patience = 3
    else:
        patience -= 1

    if patience <= 0: # Early stopping with patience
        break # Not nice but it keeps the code clean

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print('Test ppl: ', final_ppl)

path = "./output/model.pt"
torch.save(model.state_dict(), path)
model = LM_RNN(emb_size, hid_size, len(lang), pad_index=pad_token_id).to(device)
model.load_state_dict(torch.load(path))