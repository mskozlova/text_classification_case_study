from collections import defaultdict
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchtext.data import get_tokenizer


tokenizer = get_tokenizer("basic_english")


class CorpusDictionary:
    def __init__(self, texts, unknown_symbol="UNK", pad_symbol="PAD"):
        tokenized_texts = list(map(tokenizer, texts))
        
        self.dictionary = self._create_dictionary(tokenized_texts)
        self.n_texts = len(texts)
        self.unknown_symbol = unknown_symbol
        self.pad_symbol = pad_symbol
        
        self.word_to_idx, self.idx_to_word = self._create_indexers(unknown_symbol, pad_symbol)

  
    def _create_dictionary(self, tokenized_texts):
        dictionary = defaultdict(int)
        for tokens in tokenized_texts:
            tokens = set(tokens)
            for token in tokens:
                dictionary[token] += 1
        
        return dictionary


    def _create_indexers(self, unknown_symbol, pad_symbol):
        word_to_idx = {word: idx for word, idx in zip(self.dictionary.keys(), range(2, len(self.dictionary) + 2))}
        word_to_idx[pad_symbol] = 0
        word_to_idx[unknown_symbol] = 1

        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        return word_to_idx, idx_to_word
    
    
    def get_frequencies(self):
        return sorted(
            [(word, frequency / self.n_texts) for word, frequency in self.dictionary.items()],
            key=lambda x: x[1]
        )
    
    
