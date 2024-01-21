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


def fit(model, train_dl, validation_dl, optimizer, scheduler, loss_fn, device, epochs=4):
     
    for epoch in range(0, epochs):
        print("{}\tEpoch {} / {}\nTraining...".format(datetime.datetime.now(), epoch + 1, epochs))

        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dl):

            if step % 40 == 0 and not step == 0:
                print("{}\tstep {} / {} done".format(datetime.datetime.now(), step, len(train_dl)))

            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)
                    
            model.zero_grad()     
            
            output = model(
                ids=input_ids,
                mask=input_mask,
                token_type_ids=None
            )

            optimizer.zero_grad()
            loss = loss_fn(output, labels)
            
            total_train_loss += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dl)
                
        print("{}\tAverage training loss: {:.2f}".format(datetime.datetime.now(), avg_train_loss))
            
        print("")
        print("{}\tValidating...".format(datetime.datetime.now()))

        model.eval()

        total_eval_loss = 0

        for batch in validation_dl:
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            with torch.no_grad():
                output = model(
                    ids=input_ids,
                    mask=input_mask,
                    token_type_ids=None
                )
                
            loss = loss_fn(output, labels)
            total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(validation_dl)
        
        print("{}\tAverage validation loss: {:.2f}".format(datetime.datetime.now(), avg_val_loss))
        print()

    print()
    print("Training complete!")


def predict(model, dl, device):
    model.eval()

    predictions, true_labels = [], []

    for batch in dl:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
    
        with torch.no_grad():
            output = model(
                ids=input_ids,
                mask=input_mask,
                token_type_ids=None
            )

        probs = F.softmax(output, dim=1).detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
    
        predictions.extend(list(probs))
        true_labels.extend(list(label_ids))

    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)

    return true_labels, predictions
