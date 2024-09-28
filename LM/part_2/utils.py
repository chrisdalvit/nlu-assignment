import math
import json
from functools import partial

import torch 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class NTAvgSGD(optim.SGD):
    
    def __init__(self, model, eval_data, eval_criterion, lr, L, n) -> None:
        super(NTAvgSGD, self).__init__(model.parameters(), lr=lr)
        self._model = model
        self._eval_data = eval_data
        self._eval_criterion = eval_criterion
        self._lr = lr
        self._L = L
        self._n = n 
        self._params_sum = dict()
        self._k = 0
        self._t = 0 
        self._T = 0
        self._logs = []
        
    def step(self):
        super(NTAvgSGD, self).step()
        with torch.no_grad():
            if self._k % self._L == 0 and self._T == 0:
                v = self._compute_val_perpelexity()
                log_idx = self._t - self._n - 1
                min_log = min(self._logs[:log_idx]) if log_idx > 0 else float("inf")
                if self._t > self._n and v > min_log:
                    self._T = self._k
                self._logs.append(v)
                self._t += 1
            self._k += 1
            
        if self._T > 0: # If averaging was triggered
            for p in self.param_groups[0]['params']: # Compute Sum_T^k w_i
                if p in self._params_sum:
                    self._params_sum[p] = p.data.clone()
                else:
                    self._params_sum[p] += p.data
        
    def update_weights(self):
        if self._T > 0:
            for p in self.param_groups[0]['params']:
                p.data = self._params_sum[p].clone() / (self._k - self._T + 1)

    # Maybe make this function to parameter
    def _compute_val_perpelexity(self):
        self._model.eval()
        loss_array = []
        number_of_tokens = []
        with torch.no_grad():
            for sample in self._eval_data:
                output = self._model(sample['source'])
                loss = self._eval_criterion(output, sample['target'])
                loss_array.append(loss.item())
                number_of_tokens.append(sample["number_tokens"])

        ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
        return ppl
    

class Lang():
    
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
        
    def get_vocab(self, corpus, special_tokens=[]):
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
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res
    
    
class Logger:
    
    def __init__(self, env) -> None:
        self.data = {
            "args": vars(env.args),
            "epochs": [],
            "final_perplexity": None
        }
    
    def add_epoch_log(self, epoch, train_loss, eval_loss, ppl):
        self.data["epochs"].append({ 
            "epoch": epoch,
            "train_loss": train_loss, 
            "eval_loss": eval_loss, 
            "perplexity": ppl
        })
        
    def set_final_ppl(self, ppl):
        self.data["final_perplexity"] = ppl    
    
    def dumps(self):
        return json.dumps(self.data)
    
    
class Environment:
    
    def __init__(
        self,
        args,
        train_path="../dataset/PennTreeBank/ptb.train.txt",
        dev_path="../dataset/PennTreeBank/ptb.valid.txt",
        test_path="../dataset/PennTreeBank/ptb.test.txt",
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
        

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
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
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

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