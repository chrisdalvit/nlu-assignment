import json
import argparse
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from model import ModelABSA
from functions import train_loop, eval_loop
from utils import load_data, Lang, SemEval, collate_fn, _compute_lang_data

BERT_VERSIONS = [
    "bert-base-uncased", 
    "bert-large-uncased", 
    "bert-tiny-uncased", 
    "bert-small-uncased", 
    "bert-medium-uncased", 
    "bert-mini-uncased"
]

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--clip", type=float, default=5)
parser.add_argument("--num-epochs", type=int, default=3)
parser.add_argument("--train-batch-size", type=float, default=32)
parser.add_argument("--dropout", type=float, default=0.1)
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
    version=args.bert_version
    
    raw_train_data = load_data("dataset_processed/laptop14_train.json")
    raw_test_data = load_data("dataset_processed/laptop14_test.json")
    
    slots = _compute_lang_data(raw_train_data, raw_test_data)
    lang = Lang(slots)
    out_slots = len(lang.slot2id)
    
    tokenizer = BertTokenizer.from_pretrained(version)
    bert = BertModel.from_pretrained(version)
    model = ModelABSA(tokenizer, bert, out_slots, dropout).to(device)
    
    train_dataset = SemEval(raw_train_data, lang)
    test_dataset = SemEval(raw_test_data, lang)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, lang=lang, device=device))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, lang=lang, device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss(ignore_index=pad_token)

    for _ in range(num_epochs):
        train_loss_array = train_loop(model, train_dataloader, optimizer, criterion, device, clip)       
        eval_loss_array, f1 = eval_loop(model, test_dataloader, criterion, lang, device) 
        print(json.dumps({
            "train_losses": np.array(train_loss_array).mean(),
            "eval_losses": np.array(eval_loss_array).mean(),
            "f1": f1
        }))
       
if __name__ == "__main__":
    main()