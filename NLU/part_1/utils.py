import json
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class Environment:
    """Utility class for storage of hyperparameters and training run configuration."""
    
    def __init__(
        self, 
        args,
        train_path="./dataset/ATIS/train.json",
        test_path="./dataset/ATIS/test.json",
        portion=0.1
    ) -> None:
        """Init environment.

        Args:
            args: Parsed command line arguments.
            train_path (str, optional): Path to training data. Defaults to "./dataset/ATIS/train.json".
            test_path (str, optional): Path to test data. Defaults to "./dataset/ATIS/test.json".
            portion (float, optional): Train and Evaluation data ratio. Defaults to 0.1.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pad_token = 0
        self.args = args
        self.portion = portion
        tmp_train_raw = load_data(train_path)
        self.test_raw = load_data(test_path)
        inputs, labels, mini_train = self._count_intents(tmp_train_raw)
        self.train_raw, self.dev_raw, _, _ = train_test_split(inputs, labels, test_size=self.portion, random_state=42, shuffle=True, stratify=labels)
        self.train_raw.extend(mini_train)
        self.words, self.intents, self.slots = self._compute_lang_data(self.train_raw, self.dev_raw, self.test_raw)
        
    def _count_intents(self, data):
        intents = [x['intent'] for x in data]
        count_y = Counter(intents)
        inputs, labels, mini_train = [], [], []
        for id_y, y in enumerate(intents):
            if count_y[y] > 1:
                inputs.append(data[id_y])
                labels.append(y)
            else:
                mini_train.append(data[id_y])
        return inputs, labels, mini_train

    def _compute_lang_data(self, train_raw, dev_raw, test_raw):
        words = sum([x['utterance'].split() for x in train_raw], [])
        corpus = train_raw + dev_raw + test_raw
        slots = set(sum([line['slots'].split() for line in corpus],[]))
        intents = set([line['intent'] for line in corpus])
        return words, intents, slots
    
    @property
    def lang(self):
        return Lang(self.words, self.intents, self.slots, cutoff=0)
    
class Logger:
    """Class for custom data logger."""
    
    def __init__(self, env) -> None:
        self.data = {
            "args": vars(env.args),
            "runs": []
        }
        
    def add_run(self):
        """Add record for new run."""
        self.data["runs"].append({
            "epochs": [],
            "final_slot_f1": None,
            "final_intent_accuracy": None
        })
    
    def add_epoch_log(self, run, epoch, train_loss, eval_loss, f1):
        """Add record for single epoch."""
        self.data["runs"][run]["epochs"].append({ 
            "epoch": epoch,
            "train_loss": train_loss, 
            "eval_loss": eval_loss, 
            "f1": f1
        })
        
    def set_final_scores(self, run, slot_f1, intent_acc):
        """Set final test scores of training run."""
        self.data["runs"][run]["final_slot_f1"] = slot_f1
        self.data["runs"][run]["final_intent_accuracy"] = intent_acc    
    
    def dumps(self):
        """Dump log data to JSON string."""
        return json.dumps(self.data)

class Lang():
    """Utility class for computation and storage of vocabulary. Class taken from Lab 5 (Intent Classification and Slot Filling)"""
    def __init__(self, words, intents, slots, cutoff=0, pad_token=0):
        self.pad_token = pad_token
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        """Compute word to id mapping of vocabulary."""
        vocab = {'pad': self.pad_token}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots(Dataset):
    """Class for ATIS dataset. Class taken from Lab 5 (Intent Classification and Slot Filling)"""
    
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
        
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper):
        """Map sequences of tokens to corresponding computed in Lang class."""
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
def collate_fn(data, pad_token, device):
    """Function applied to batches. Function taken from Lab 5 (Intent Classification and Slot Filling)"""
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset