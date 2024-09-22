import torch
from torch.utils.data import DataLoader

from .Lang import Lang
from .utils import read_file, get_vocab, collate_fn
from dataset.PennTreeBank import PennTreeBank

TRAIN_DATASET_PATH = "dataset/PennTreeBank/ptb.train.txt"
DEV_DATASET_PATH = "dataset/PennTreeBank/ptb.valid.txt"
TEST_DATASET_PATH = "dataset/PennTreeBank/ptb.test.txt"

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"

class Environment:
    
    def __init__(
        self,
        train_batch_size=64,
        dev_batch_size=128,
        test_batch_size=128,
        train_path=TRAIN_DATASET_PATH,
        dev_path=DEV_DATASET_PATH,
        test_path=TEST_DATASET_PATH,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_raw = read_file(train_path)
        self.dev_raw = read_file(dev_path)
        self.test_raw = read_file(test_path)
        self.vocab = get_vocab(train_raw, [pad_token, eos_token])

        self.lang = Lang(train_raw, [pad_token, eos_token])
        self.pad_token_id = self.ang.word2id[pad_token]

        self.train_dataset = PennTreeBank(self.train_raw, self.lang)
        self.dev_dataset = PennTreeBank(self.dev_raw, self.lang)
        self.test_dataset = PennTreeBank(self.test_raw, self.lang)

        # Dataloader instantiation
        # You can reduce the batch_size if the GPU memory is not enough
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device), shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=dev_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device))
        self.test_loader = DataLoader(self.test_dataset, batch_size=test_batch_size, collate_fn=partial(collate_fn, pad_token=self.pad_token_id, device=self.device))
