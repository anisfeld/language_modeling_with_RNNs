import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

import os
import torch
from collections import Counter

class Dictionary2(object):
    def __init__(self):
        self.word2idx = Counter()

    def counts_to_idx(self):
        '''order vocab -- useful for adaptive softmax'''
        '''adapted from https://github.com/rosinality/adaptive-softmax-pytorch '''
        vocab = sorted(self.word2idx.items(), key=lambda x: x[1], reverse=True)

        self.word2idx = {k: id for id, (k, _) in enumerate(vocab)}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx) -1

    def __len__(self):
        return len(self.word2idx)


class Corpus2(object):
    def __init__(self, path):
        self.dictionary = Dictionary2()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))


    def tokenize(self, path, train=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    if train:
                        self.dictionary.word2idx[word] += 1
                    else:
                        self.dictionary.add_word(word)

        if train:
            self.dictionary.counts_to_idx()


        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
        