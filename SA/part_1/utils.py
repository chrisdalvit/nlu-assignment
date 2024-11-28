import json

import torch
from torch.utils.data import Dataset

def _compute_lang_data(train_raw, test_raw):
    corpus = train_raw + test_raw
    slots = set(sum([line['slots'] for line in corpus],[]))
    return slots

def apply_first_subtoken_strategy(inputs, tokenizer):
    filtered_inputs = []
    for sentence in inputs:
        first_token_sentence = " ".join(tokenizer.tokenize(word)[0] for word in sentence.split())
        filtered_inputs.append(first_token_sentence)
    return filtered_inputs

def collate_fn(data, tokenizer, lang, device):
    fst_subtokens = []
    for sample in data:
        fst_subtokens.append(apply_first_subtoken_strategy(sample['tokens'], tokenizer))
    
    max_slots_len = max(len(x['slots']) for x in data)
    padded_slots = []
    for sample in data:
        diff = max_slots_len - len(sample['slots'])
        pad = [lang.pad_token] * diff
        padded_slots.append(sample['slots'].tolist() + pad)
    X = tokenizer(fst_subtokens, padding=True, is_split_into_words=True, return_tensors="pt")
    y = torch.LongTensor(padded_slots)    
    return X.to(device), y.to(device), data

class Lang():
    
    def __init__(self, slots, pad_token=0):
        self.pad_token = pad_token
        self.slot2id = self.lab2id(slots)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['[PAD]'] = self.pad_token # Use same pad token as BERT
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class SemEval(Dataset):
    
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.slots = []
        self.tokens = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['sentence'])
            self.tokens.append(x['tokens'])
            self.slots.append(x['slots'])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        slots = torch.Tensor(self.slot_ids[idx])
        return {'slots': slots, 'tokens': self.tokens[idx] }
    
    def mapping_seq(self, data, mapper):
        """Map sequences of tokens to corresponding computed in Lang class."""
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset