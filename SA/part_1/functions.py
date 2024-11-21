import torch 

def train_loop(model, dataloder, optimizer, criterion, device, clip):
    model.train()
    loss_array = []
    for X, y, samples in dataloder:
        optimizer.zero_grad()
        slots = model(X)
        loss = criterion(slots.to(device), y)
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
    return loss_array

def eval_loop(model, dataloader, criterion, num_classes, device):
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
            
            metric_inputs = []
            for idx, slot in enumerate(slots):
                pred = slot.softmax(dim=0).argmax(dim=0)
                slot_len = len(samples[idx]['tokens'])
                metric_inputs.append((y[idx][:slot_len], pred[:slot_len]))
                
        return loss_array, evaluate(metric_inputs, num_classes)
    
def evaluate(metrics, num_classes, eps=1e-8):
    tp, fp, fn = 0, 0, 0
    for gt, pred in metrics:
        for c in range(num_classes): 
            tp += ((pred == c) & (gt == c)).sum().float()
            fp += ((pred == c) & (gt != c)).sum().float()
            fn += ((pred != c) & (gt == c)).sum().float()
            
    # Compute macro and micro averages
    precision = tp / (tp + fp + eps)  # Add epsilon to avoid division by zero
    recall = tp / (tp + fn + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    return {
        "class_precision": precision.tolist(),
        "class_recall": recall.tolist(),
        "class_f1": f1_score.tolist(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1_score.mean().item(),
    }