#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 12:05:36 2021

@author: jeff
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from networks.RNN import rnn, seq_encoder_decoder, attention_decoder, transformer
from training import train_rnn, train_seq_autoencoder

'''
#%% Language model for predicting the next character in a sequence

#Number of sequences in each batch
batch_size = 32
#Number of time steps in each sequence
num_steps = 35

num_hiddens = 256
num_layers = 1
unit_type = 'gru'

#Vocab is a dictionary of characters to indices
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

if torch.cuda.is_available():
    device = torch.device('cuda')

#Show the vocab
#print(list(vocab.token_to_idx.items())[:])

#Initialize the model
net = rnn.RNNModel(len(vocab), num_hiddens, num_layers, unit_type)
net.to(device)

#Train the model
num_epochs, lr = 500, 1
train_rnn.train(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())


#Try out the model
#Try predicting the next few characters
print(train_rnn.predict('time traveller ', 50, net, vocab, d2l.try_gpu()))

'''

#%% Train and test sequential autoencoder
'''
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

#Load the data
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

#Initialize the encoder and decoders
encoder = seq_encoder_decoder.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
decoder = seq_encoder_decoder.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
net = seq_encoder_decoder.EncoderDecoder(encoder, decoder)


#Train the network
train_seq_autoencoder.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

#Evaluate the network
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = train_seq_autoencoder.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {train_seq_autoencoder.bleu(translation, fra, k=2):.3f}')

'''




#%% Train and test attention autoencoder
'''
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)


#Initiate the network
encoder = seq_encoder_decoder.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens,
                             num_layers, dropout)
decoder = attention_decoder.Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens,
                                  num_layers, dropout)
net = seq_encoder_decoder.EncoderDecoder(encoder, decoder)


#Train the network
train_seq_autoencoder.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


#Test out the network
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = train_seq_autoencoder.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

#Look at the vocab
#print(list(src_vocab.token_to_idx.items())[:])

#Get the attention weights
attention_weights = torch.cat(
    [step[0][0][0] for step in dec_attention_weight_seq], 0).reshape(
        (1, 1, -1, num_steps))
        
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')

#Note: There are 4 key positions that correspond to the i'm, home, ., <eos> (Number of encoder hidden states)
#and 6 key positions that correspond to je, suis, chez, moi, ., <eos> (Number of decoder hidden states)
'''


#%% Train and test transformer

num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

#Load data
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)


#Load the encoder and decoder of the transformer
encoder = transformer.TransformerEncoder(len(src_vocab), key_size, query_size, value_size,
                             num_hiddens, norm_shape, ffn_num_input,
                             ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = transformer.TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size,
                             num_hiddens, norm_shape, ffn_num_input,
                             ffn_num_hiddens, num_heads, num_layers, dropout)
net = seq_encoder_decoder.EncoderDecoder(encoder, decoder)
train_seq_autoencoder.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')




