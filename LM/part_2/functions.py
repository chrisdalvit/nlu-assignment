import math 

import torch

def train_loop(data, optimizer, criterion, model, clip=5):
    """Run one epoch of training. Function taken from Lab 4 (Neural Language Modelling)

    Args:
        data: Training dataloader
        optimizer: Optimizer for weight update
        criterion: Training loss function
        model: PyTorch model
        clip (int, optional): Max norm for gradient clipping. Defaults to 5.

    Returns:
        int: Average training loss of the epoch
    """
    model.train()
    loss_array = []
    number_of_tokens = []
    for sample in data:
        optimizer.zero_grad()
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    """Run one epoch of evaluation. Function taken from Lab 4 (Neural Language Modelling)

    Args:
        data: Evaluation dataloader
        eval_criterion: Evalutation loss function
        model: PyTorch model

    Returns:
        tuple: Evalutation perplexity, average evaluation loss of the epoch
    """
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    """Initialize model weights. Function taken from Lab 4 (Neural Language Modelling)

    Args:
        mat: PyTorch module
    """
    for m in mat.modules():
        if type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [torch.nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)