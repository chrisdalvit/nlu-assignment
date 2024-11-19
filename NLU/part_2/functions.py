import torch
import torch.nn as nn
from conll import evaluate
from sklearn.metrics import classification_report

def apply_first_subtoken_strategy(inputs, tokenizer):
    """Filter the input according to the first subtoken strategy. 
    
    The inputs are tokenized and only the first subtoken for each word is stored. All other subtokens are discarded.

    Args:
        inputs: Batch of sentences.
        tokenizer: Tokenizer.

    Returns:
        list: Batch of tokenized sentences
    """
    filtered_inputs = []
    for sentence in inputs:
        first_token_sentence = " ".join(tokenizer.tokenize(word)[0] for word in sentence.split())
        filtered_inputs.append(first_token_sentence)
    return filtered_inputs


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, env, clip=5):
    """Run one epoch of training. Function taken from Lab 5 (Intent Classification and Slot Filling)

    Args:
        data: Train dataloader.
        optimizer: Train optimizer.
        criterion_slots: Loss function for slot filling.
        criterion_intents: Loss function for intent classification.
        model: PyTorch model.
        env: Training environment.
        clip (int, optional): Gradient clipping. Defaults to 5.

    Returns:
        list: Array of train losses
    """
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        inputs = apply_first_subtoken_strategy(sample['sentence'], model.tokenizer)
        slots, intent = model(inputs)
        
        loss_intent = criterion_intents(intent.to(env.device), sample['intents'])
        loss_slot = criterion_slots(slots.to(env.device), sample['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, env):
    """Run one epoch of evaluation. Function taken from Lab 5 (Intent Classification and Slot Filling)

    Args:
        data: Evaluation dataloader.
        criterion_slots: Loss function for slot filling.
        criterion_intents: Loss function for intent classification.
        model: PyTorch model.
        env: Training environment.

    Returns:
        tuple: Evaluation metrics.
    """
    model.eval()
    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []
    with torch.no_grad():
        for sample in data:
            inputs = apply_first_subtoken_strategy(sample['sentence'], model.tokenizer)
            slots, intents = model(inputs)
            loss_intent = criterion_intents(intents.to(env.device), sample['intents'])
            loss_slot = criterion_slots(slots.to(env.device), sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            out_intents = [env.lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [env.lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [env.lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [env.lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], env.lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array
