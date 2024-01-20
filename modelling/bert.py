import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from transformers import BertModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def get_dataset(texts, target, max_vector_len=50):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = max_vector_len,
            truncation=True,
            padding="max_length",
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(target).type(torch.float)
        
    return TensorDataset(input_ids, attention_masks, labels)
