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
    
    
    def truncate_dictionary(self, min_frequency=0.0, max_frequency=1.0):
        self.dictionary = {
            word: frequency
            for word, frequency in self.dictionary.items()
            if frequency / self.n_texts >= min_frequency
                and frequency / self.n_texts <= max_frequency
        }
        self.word_to_idx, self.idx_to_word = self._create_indexers(self.unknown_symbol, self.pad_symbol)
    
    
    def transform(self, texts):
        tokenized_texts = list(map(tokenizer, texts))
        
        return [
            [
                self.word_to_idx.get(
                    token,
                    self.word_to_idx[self.unknown_symbol]
                ) for token in tokens
            ] for tokens in tokenized_texts
        ]


class PaddedTextVectorDataset(Dataset):
    def __init__(self, texts, target, corpus_dict=None, emb=None, max_vector_len=50):
        self.max_vector_len = max_vector_len
        vectors = self._get_vectors(texts, corpus_dict, emb)
        self.lengths = list(map(
            lambda x: len(x) if len(x) <= max_vector_len else max_vector_len,
            vectors
        ))
        self.vectors = list(map(self.pad_data, vectors))
        self.target = list(target)
        
        assert len(texts) == len(target), "Texts len != target len: {} != {}".format(len(texts), len(target))


    def _get_vectors(self, texts, corpus_dict, emb):
        if corpus_dict is not None:
            assert emb is None, "Can't provide both corpus_dict and pretrained embeddings"
            return corpus_dict.transform(texts)
        
        if emb is not None:
            assert corpus_dict is None, "Can't provide both corpus_dict and pretrained embeddings"
            tokenized_texts = list(map(tokenizer, texts))
            
            vectors = [[emb.stoi[token] for token in tokens if token in emb.stoi] for tokens in tokenized_texts]
            return [vector if len(vector) > 0 else [0] for vector in vectors]

        raise ValueError("Should provide one of: corpus_dict, pretrained embeddings")


    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        X = np.asarray(self.vectors[idx])
        vector_len = np.asarray(self.lengths[idx])
        y = np.asarray(self.target[idx]).astype(float)
        return X, y, vector_len


    def pad_data(self, s):
        padded = np.zeros((self.max_vector_len,), dtype=np.int64)
        if len(s) > self.max_vector_len:
            padded[:] = s[:self.max_vector_len]
        else:
            padded[:len(s)] = s
        return padded


def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0, 1), y, lengths


def fit(model, train_dl, test_dl, loss_fn, opt, epochs=3):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss_train = 0
        total_loss_test = 0
        
        for X, y, lengths in iter(train_dl):
            X, y, lengths = sort_batch(X, y, lengths)
            X = Variable(X)
            y = Variable(y)
            lengths = lengths.numpy()
            
            opt.zero_grad()
            pred = model(X, lengths)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            
            total_loss_train += loss.item()

        train_loss = total_loss_train / len(train_dl)
        train_losses.append(train_loss)
        
        if epoch % 5 == 0:
            model.eval()
            for X, y, lengths in test_dl:
                X, y, lengths = sort_batch(X, y, lengths)
                X = Variable(X)
                y = Variable(y)
                pred = model(X, lengths.numpy())
                loss = loss_fn(pred, y)
                total_loss_test += loss.item()

            test_loss = total_loss_test / len(test_dl)
            test_losses.append(test_loss)

        clear_output(wait=True)
        
        print("Train loss:\t{:.4f}\nVal loss:\t{:.4f}".format(train_loss, test_losses[-1]))
        
        plt.figure(figsize=(8, 4))
        plt.plot(range(epoch + 1), train_losses, label='train', marker='o')
        plt.plot(range(0, epoch + 1, 5), test_losses, label='test', marker='o')
        plt.legend()
        plt.title("Training loss")
        plt.show()
   
    return train_losses, test_losses


def predict(model, dl):
    y_true_val, y_pred_val, y_pred_proba = [], [], []

    model.eval()
    for X, y, lengths in dl:
        X, y, lengths = sort_batch(X, y, lengths)
        X = Variable(X)
        y = Variable(y)
        pred = model(X, lengths.numpy())
        prob = F.softmax(pred, dim=1)
        pred_idx = torch.max(pred, 1)[1]
        y_true_val += list(y.cpu().data.numpy())
        y_pred_val += list(pred_idx.cpu().data.numpy())
        y_pred_proba += list(prob.cpu().data.numpy())
        
    return np.asarray(y_true_val), np.asarray(y_pred_val), np.asarray(y_pred_proba)
