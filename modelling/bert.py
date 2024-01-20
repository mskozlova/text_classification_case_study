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


class BERTModel(nn.Module):
    def __init__(self, n_out=12):
        super(BERTModel, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = nn.Dropout(0.3)
        self.l3 = nn.Linear(768, n_out)


    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1.pooler_output)
        output = self.l3(output_2)
        return output
