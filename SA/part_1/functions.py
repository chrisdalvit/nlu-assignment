import torch 
from sklearn.metrics import f1_score

from utils import Lang

def train_loop(model, dataloder, optimizer, criterion, device, clip):
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
    model.eval()
    loss_array = []
    ys = []
    slots_array = []
    with torch.no_grad():
        for X, y, samples in dataloader:
            slots = model(X)
            loss = criterion(slots.to(device), y)
            loss_array.append(loss.item())
            ys.extend(y)
            slots_array.append(slots)
            
            gt_array = []
            pred_array = []
            for idx, slot in enumerate(slots):
                pred = slot.softmax(dim=0).argmax(dim=0)
                slot_len = len(samples[idx]['tokens'])
                gt_array.append(y[idx][:slot_len].cpu())
                pred_array.append(pred[:slot_len].cpu())
                
        f1s = f1_score(gt_array, pred_array, average=None)
        return loss_array, {v: f1 for v, f1 in zip(lang.slot2id.keys(), f1s)}