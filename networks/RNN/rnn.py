#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:50:09 2021

@author: jeff
"""
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l




class RNNModel(nn.Module):
    """The RNN model."""
    def __init__(self, vocab_size, num_hidden, num_layers = 1, unit_type = 'rnn', **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        
        #Define the type of recurrent unit to use
        if unit_type == 'rnn':
            self.rnn = nn.RNN(vocab_size, num_hidden, num_layers)
        elif unit_type == 'gru':
            self.rnn = nn.GRU(vocab_size, num_hidden, num_layers)
        elif unit_type == 'lstm':
            self.rnn = nn.LSTM(vocab_size, num_hidden, num_layers)
        else:
            RuntimeError('Unit type not supported. Options are rnn, gru, lstm')
        
        #Input size is the vocab size
        self.vocab_size = vocab_size
        
        #Number of hidden units
        self.num_hiddens = self.rnn.hidden_size
        
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))