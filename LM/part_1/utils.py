import json
from functools import partial

import torch 
from torch.utils.data import DataLoader, Dataset


class Lang():
    """Utility class for computation and storage of vocabulary. Class taken from Lab 4 (Neural Language Modelling)"""
    
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
        
    def get_vocab(self, corpus, special_tokens=[]):
        """Compute word to id mapping of vocabulary."""
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    
    def __len__(self):
        return len(self.word2id)
    
    
class PennTreeBank(Dataset):
    """Class for PennTreeBank dataset. Class taken from Lab 4 (Neural Language Modelling)"""

    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])
            self.target.append(sentence.split()[1:])
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    def mapping_seq(self, data, lang):
        """Map sequences of tokens to corresponding computed in Lang class."""
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res
    
    
class Logger:
    """Class for custom data logger."""
    
    def __init__(self, env) -> None:
        self.data = {
            "args": vars(env.args),
            "epochs": [],
            "final_perplexity": None
        }
    
    def add_epoch_log(self, epoch, train_loss, eval_loss, ppl):
        """Add record for single epoch."""
        self.data["epochs"].append({ 
            "epoch": epoch,
            "train_loss": train_loss, 
            "eval_loss": eval_loss, 
            "perplexity": ppl
        })
        
    def set_final_ppl(self, ppl):
        """Set final test perplexity of training run."""
        self.data["final_perplexity"] = ppl    
    
    def dumps(self) -> str:
        """Dump log data to JSON string."""
        return json.dumps(self.data)
    
    
class Environment:
    """Utility class for storage of hyperparameters and training run configuration."""
    
    def __init__(
        self,
        args,
        train_path="../dataset/PennTreeBank/ptb.train.txt",
        dev_path="../dataset/PennTreeBank/ptb.valid.txt",
        test_path="../dataset/PennTreeBank/ptb.test.txt",
        pad_token="<pad>",
        eos_token="<eos>"
    ):
        """Init environment.

        Args:
            args: Parsed command line arguments.
            train_path (str, optional): Path to training data. Defaults to "../dataset/PennTreeBank/ptb.train.txt".
            dev_path (str, optional): Path to evaluation data. Defaults to "../dataset/PennTreeBank/ptb.valid.txt".
            test_path (str, optional): Path to test data. Defaults to "../dataset/PennTreeBank/ptb.test.txt".
            pad_token (str, optional): Padding token. Defaults to "<pad>".
            eos_token (str, optional): End of sentence token. Defaults to "<eos>".
        """
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
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device), shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=dev_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device))
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device))
        return {
            "train": train_loader, 
            "dev": dev_loader, 
            "test": test_loader
        }
        

def read_file(path, eos_token="<eos>"):
    """Utility function for reading dataset file. Function taken from Lab 4 (Neural Language Modelling)"""
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    """Utility function for vocabulary computation. Function taken from Lab 4 (Neural Language Modelling)"""
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

def collate_fn(data, pad_token, device):
    """Function applied to batches. Function taken from Lab 4 (Neural Language Modelling)"""
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item