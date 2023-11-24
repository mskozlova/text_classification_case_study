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

