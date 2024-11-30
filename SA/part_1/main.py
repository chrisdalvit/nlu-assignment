import copy
import argparse
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

from model import ModelABSA
from functions import train_loop, eval_loop
from utils import load_data, Lang, SemEval, collate_fn, compute_lang_data, Logger

BERT_VERSIONS = [
    "bert-base-uncased", 
    "bert-large-uncased"
]

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--clip", type=float, default=5)
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--train-batch-size", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--bert-version", type=str, choices=BERT_VERSIONS, default=BERT_VERSIONS[0])


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_token = 0
    lr = args.lr
    clip = args.clip
    num_epochs = args.num_epochs
    batch_size = args.train_batch_size
    dropout = args.dropout
    
    raw_train_data = load_data("dataset_processed/laptop14_train.json")
    raw_test_data = load_data("dataset_processed/laptop14_test.json")
    train_raw, dev_raw = train_test_split(raw_train_data, test_size=0.1, random_state=42, shuffle=True)
    
    logger = Logger(args)
    slots = compute_lang_data(raw_train_data, raw_test_data)
    lang = Lang(slots)
    out_slots = len(lang.slot2id)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_version)
    lm = BertModel.from_pretrained(args.bert_version)
    model = ModelABSA(lm, out_slots, dropout).to(device)
    
    train_dataset = SemEval(train_raw, lang)
    dev_dataset = SemEval(dev_raw, lang)
    test_dataset = SemEval(raw_test_data, lang)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, lang=lang, device=device))
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, lang=lang, device=device))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, lang=lang, device=device))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss(ignore_index=pad_token)

    best_score = 0
    best_model = None
    patience = 5
    for i in range(num_epochs):
        train_loss_array = train_loop(model, train_dataloader, optimizer, criterion, device, clip)       
        eval_loss_array, results = eval_loop(model, dev_dataloader, criterion, lang, device) 
        if results['sentiment']['f1'] > best_score:
            best_score = results['sentiment']['f1']
            best_model = copy.deepcopy(model).to(device)
            patience = 5
        else:
            patience -= 1
        if patience <= 0:
            break 
            
        logger.add_epoch_log(i, np.array(train_loss_array).mean(), np.array(eval_loss_array).mean(), results)
    
    test_loss_array, results = eval_loop(best_model, test_dataloader, criterion, lang, device)
    logger.set_final_scores(np.array(test_loss_array).mean(), results)
    print(logger.dumps())
    
if __name__ == "__main__":
    main()
