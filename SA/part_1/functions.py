import torch 

from utils import Lang
from evals import evaluate_ote

def train_loop(model, dataloder, optimizer, criterion, device, clip):
    """Run one epoch of training. Function taken from Lab 5 (Intent Classification and Slot Filling)

    Args:
        model: PyTorch model.
        dataloader: Train dataloader.
        optimizer: Train optimizer.
        criterion: Loss function.
        device: PyTorch device.
        clip (int, optional): Gradient clipping.

    Returns:
        list: Array of train losses
    """
    model.train()
    loss_array = []
    for X, y, _ in dataloder:
        optimizer.zero_grad()
        slots = model(X)
        loss = criterion(slots.to(device), y)
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
    return loss_array

def eval_loop(model, dataloader, criterion, lang: Lang, device):
    """Run one epoch of evaluation. Function taken from Lab 5 (Intent Classification and Slot Filling)

    Args:
        model: PyTorch model.
        dataloader: Evaluation dataloader.
        criterion: Loss function.
        lang: Lang object used for evaluation.
        device: PyTorch device.

    Returns:
        tuple: Evaluation metrics.
    """
    model.eval()
    loss_array = []
    ys = []
    slots_array = []
    gt_array = []
    pred_array = []
    with torch.no_grad():
        for X, y, samples in dataloader:
            slots = model(X)
            loss = criterion(slots.to(device), y)
            loss_array.append(loss.item())
            ys.extend(y)
            slots_array.append(slots)
            
            preds = slots.argmax(dim=1)
            for idx, pred in enumerate(preds):
                slot_len = len(samples[idx]['tokens'])
                gt_array.append(y[idx][:slot_len].cpu())
                pred_array.append(pred[:slot_len].cpu())
                
        try:
            results = evaluate_ote(gt_array, pred_array, lang)
        except Exception as ex:
            print("Warning:", ex)
            results = {'precision': 0, 'recall': 0, 'f1': 0}
        return loss_array, results