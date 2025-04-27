# dataset.py

from tokenizers import Tokenizer
import torch
from config import config

class BpeDataset:
    def __init__(self, filepath):
        self.tokenizer = Tokenizer.from_file("tokenizer.json")

        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.ids = torch.tensor(self.tokenizer.encode(self.text).ids, dtype=torch.long)
        n = int(len(self.ids) * config["train_test_split"])
        self.train_data = self.ids[:n]
        self.val_data = self.ids[n:]
        self.vocab_size = self.tokenizer.get_vocab_size()

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(0, len(data) - config["block_size"], (config["batch_size"],))
        x = torch.stack([data[i:i+config["block_size"]] for i in ix])
        y = torch.stack([data[i+1:i+1+config["block_size"]] for i in ix])
        return x.to(config["device"]), y.to(config["device"])

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
