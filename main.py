import argparse
import copy

import torch
import torch.optim as optim

from utils.environment import Environment
from utils.utils import train_loop, eval_loop
from models import get_model, save_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="rnn")
# Experiment also with a smaller or bigger model by changing hid and emb sizes
# A large model tends to overfit
parser.add_argument("--hid-size", type=int, default=200)
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


def run_epochs(model, optimizer, criterion_train, criterion_eval, env: Environment, patience=3):
    best_ppl = float('inf')
    best_model = None
    current_patience = patience
    for epoch in range(env.args.epochs):
        loss_train = train_loop(env.dataloaders["train"], optimizer, criterion_train, model, env.args.clip)
        #losses_train.append(np.asarray(loss).mean())
        ppl_dev, loss_dev = eval_loop(env.dataloaders["dev"], criterion_eval, model)
        #losses_dev.append(np.asarray(loss_dev).mean())
        print("Train loss: ", loss_train, " Dev loss: ", loss_dev)
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            current_patience = patience
        else:
            current_patience -= 1
        if current_patience <= 0:
            break
        
    best_model.to(env.device)
    return best_model

def main():
    args = parser.parse_args()
    env = Environment(args)
    model = get_model(env)
    optimizer = optim.SGD(model.parameters(), lr=env.args.lr)
    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=env.pad_token_id)
    criterion_eval = torch.nn.CrossEntropyLoss(ignore_index=env.pad_token_id, reduction='sum')
    best_model = run_epochs(model, optimizer, criterion_train, criterion_eval, env)
    final_ppl,  _ = eval_loop(env.dataloaders["test"], criterion_eval, best_model)
    print('Final PPL: ', final_ppl)
    #save_model(best_model)

if __name__ == "__main__":
    main()