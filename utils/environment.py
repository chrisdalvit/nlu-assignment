from functools import partial

import torch
from torch.utils.data import DataLoader

from .Lang import Lang
from .utils import read_file, get_vocab, collate_fn
from dataset.PennTreeBank import PennTreeBank

class Environment:
    
    def __init__(
        self,
        args,
        train_path="dataset/PennTreeBank/ptb.train.txt",
        dev_path="dataset/PennTreeBank/ptb.valid.txt",
        test_path="dataset/PennTreeBank/ptb.test.txt",
        pad_token="<pad>",
        eos_token="<eos>"
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_raw = read_file(train_path)
        self.dev_raw = read_file(dev_path)
        self.test_raw = read_file(test_path)
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.args = args
        
    @property
    def pad_token_id(self):
        return self.lang.word2id[self.pad_token]

    @property
    def lang(self):
        return Lang(self.train_raw, [self.pad_token, self.eos_token])
    
    @property
    def vocab(self):
        return get_vocab(self.train_raw, [self.pad_token, self.eos_token])
    
    def _get_datasets(self):
        train_dataset = PennTreeBank(self.train_raw, self.lang)
        dev_dataset = PennTreeBank(self.dev_raw, self.lang)
        test_dataset = PennTreeBank(self.test_raw, self.lang)        
        return train_dataset, dev_dataset, test_dataset
    
    @property
    def dataloaders(self):
        train_batch_size, dev_batch_size, test_batch_size = self.args.train_batch_size, self.args.dev_batch_size, self.args.test_batch_size
        train_dataset, dev_dataset, test_dataset = self._get_datasets()
        # Dataloader instantiation
        # You can reduce the batch_size if the GPU memory is not enough
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device), shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=dev_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device))
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device))
        return {
            "train": train_loader, 
            "dev": dev_loader, 
            "test": test_loader
        }