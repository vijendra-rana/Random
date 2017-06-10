from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

import time
import math

class EncoderRNN(nn.Module):
    '''Encoder class for seq2seq'''
    def __init__(self, input_size,batch_size,hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size) #it will have output as (batch_size,time_steps,embedding_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #input is expected to be (batch_size,input_seq)
        embedded = self.embedding(input).view(-1, self.batch_size, self.hidden_size)
        #print(embedded.size())
        output = embedded
        for i in range(self.n_layers):
            #shape -- output - (time_step,batch,hidden_size)
            #shape -- hidden - (1,batch_size,hidden_size)
            output, hidden = self.gru(output, hidden) 
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)) #here 
        if use_cuda:
            return result.cuda()
        else:
            return result

MAX_LENGTH = 10

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,batch_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden,encoder_outputs):
        '''
        here input - (batch_size,1) -- one word at a time unlike encoder where we can pass all info. at once per our will
        hidden - (1,batch_size,hidden_size)
        the max_len below is crucial whatever we get from the encoder we have to convert that to max_len dimension
        encoder_outputs - (max_len,batch_size,hidden_size) - this is obtained after softmax on it - here we only get max_len
        '''
        
        #decoder_sequence will be given one at a time 
        print (type(input))
        embedded = self.embedding(input).view(1, batch_size, -1) #(1,batch_size,hidden_size)
        embedded = self.dropout(embedded)
        
        
        #--torch.cat - will result in  shape (batch,embedding_Size+hidden_size)
        #and then beacuse of self.attn - we get attn_weights = (batch,max_len)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        
        #At this point it is essential for bmm to have encoder_outputs to have (batch_size,max_len,hidden_size)
        encoder_outputs = encoder_outputs.transpose(1,0)
        
        
        attn_weights = attn_weights.unsqueeze(1) #shape (batch_size,1,max_length)
         
        
        attn_applied = torch.bmm(attn_weights,encoder_outputs) #(batch_size,1,hidden_size) 
        
        attn_applied = attn_applied.transpose(0,1) #(1, batch_size,hidden_size)

        output = torch.cat((embedded[0], attn_applied[0]), 1) #(batch_size,2*hidden_size)
        
        output = self.attn_combine(output).unsqueeze(0) #(1,batch_size,hidden_size)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        #finally the output is (batch_size,output_size)    
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def fake_data(no_of_pairs,max_size_len = 9,max_vocab = 10):
    #max_vocab - 10 means 0 to 9
    #1 right now is saved for eos,0 is for pad ,2 for sos 
    encoding_sentences,decoding_sentences =[],[]
    for i in range(no_of_pairs):
        encoding_sentences.append([random.randint(3,max_vocab - 1) for num in range(random.randint(1,max_size_len))])
        decoding_sentences.append([random.randint(3,max_vocab - 1) for num in range(random.randint(1,max_size_len))])
    return encoding_sentences,decoding_sentences

def batchify(pairs_sorted,batch_size):
    '''
    yields 4 Variable tensors of shape 
    encoding_sentences --(batch_size,encoding_max_len)
    encoding_mask -- (batch_size,encoding_max_len)
    decoding_sentences -- (batch_size,decoding_max_len)
    decoding_mask -- (batch_size,decoding_max_len)
    '''
    num_batches = len(pairs_sorted)//batch_size
    while True:
        for i in range(num_batches):
            pairs_result = pairs_sorted[i:i + batch_size] 
            encoding_sentences = [pair[0] for pair in pairs_result]
            #encoding sentence is already sorted so get the length of last sentence
            max_encoding_length = len(encoding_sentences[-1])
            encoding_sentences = [sentence + [0]*(max_encoding_length - len(sentence)) + [1] for sentence in encoding_sentences]
            encoding_mask = [list(map(lambda a: 0 if a == 0 else 1,sentence)) for sentence in encoding_sentences]
            
            decoding_sentences = [pair[1] for pair in pairs_result]
            max_decoding_length = len(max(decoding_sentences,key = len))
            decoding_sentences = [sentence + [0]*(max_decoding_length - len(sentence)) + [1] for sentence in decoding_sentences]
            decoding_mask = [list(map(lambda a: 0 if a == 0 else 1,sentence)) for sentence in decoding_sentences]
            yield Variable(torch.LongTensor(encoding_sentences)),Variable(torch.LongTensor(decoding_sentences)),\
                           Variable(torch.LongTensor(encoding_mask)),Variable(torch.LongTensor(decoding_mask))
        pairs_sorted = pairs_sorted[::-1] #so that each sentence is covered in 2 epochs

teacher_forcing_ratio = 0.5

def train(encoder_sentences, decoder_sentences,decoder_mask,encoder,decoder,encoder_optimizer,decoder_optimizer,max_len = MAX_LENGTH):
    '''
    Most important method -- 
    First it use the encoder forward method to get the 
        encoder_output-- shape (encoding_len,batch_size,hidden_size) 
        encoder_hidden -- shape (1,batch_size,input_size)
    
    Then crucial scenario the encoder output has to be reshaped in house from (encoding_len,batch_size,hidden_size) 
        to (max_len,batch_size,hidden_size)
    
    Second case here decoder forward method is called with 1 word at a time in return getting the 
    decoder_output-- shape (batch_size,output_size),decoder_hidden -- shape (1,batch_size,hidden_size)
    and attention weights which we will use in evaluation
    
    Finally the function returns the cross_entropy masked loss so that at least at the decoder we do not have any effect
    of padding I still have to figure out TODO masking for the encoding seq not sure how I do this
    '''
    #print('in train',type(decoder_sentences))
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #print('vijendra')
    encoder_outputs,encoder_hidden = encoder(encoder_sentences,encoder.initHidden()) #enc_output-(seq_len,batch_size,hidden_size)
    
    #now the encoder_outputs to be passed to the decoder has to be (max_len,batch_size,hidden_size) 
    
    #crucial step
    
    encoder_outputs_for_decoder = Variable(torch.zeros(max_len,batch_size,hidden_size))
    encoder_outputs_for_decoder = encoder_outputs_for_decoder.cuda() if use_cuda else encoder_outputs_for_decoder

    for i in range(output_from_encoder.size(0)):
        #print(i)
        encoder_outputs_for_decoder[i] = output_from_encoder[i]
    
    decoder_input = Variable(torch.LongTensor([2]*batch_size)).view(-1,1)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    decoding_len = decoder_sentences.size()[1]
    
    loss = 0
    #print(use_teacher_forcing)
    use_teacher_forcing = True
    if use_teacher_forcing:
        
        for i in range(decoding_len):
            #print(i)
            output_from_decoder,decoder_hidden,decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs_for_decoder)
            #As this is teacher forcing so just take the labels from the target
            #print('vijendra')
            #print(type(decoding_sentences))
            tmp = decoder_sentences[:,i].contiguous().view(-1,1)
            word_loss =  output_from_decoder.gather(1,tmp) * decoder_mask[:,i].float().contiguous().view(-1,1)
            
            loss += -1 * word_loss.sum()
            decoder_input = decoder_sentences[:,i].contiguous().view(-1,1)
    else:
        
        #No teacher forcing use its own prediction as next input
        for i in range(decoding_len):
            print(type(decoder_input))
            output_from_decoder,decoder_hidden,decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs_for_decoder)
            
            top_val, top_idx = output_from_decoder.data.topk(1)
            word_loss = output_from_decoder.gather(1,Variable(top_idx)).view(-1,1) * decoder_mask[:,i].float().contiguous().view(-1,1)
            
            loss += -1 * word_loss.sum()
            decoder_input = top_idx
    
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0]/decoding_len

def train_iters(encoder, decoder,batch_generator,batch_size, num_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    '''
    
    This function calls train function num_iters times and calculate the loss accordingly and then finally plot the results
    
    '''
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    
    num_iters = 2 * num_iters // batch_size
    
    for iter in range(1,num_iters + 1):
        
        en_sen,en_mask,dec_sen,dec_mask = next(batch_generator) #these all are variables in terms of pytorch
        print('in train iters',type(dec_sen))
        loss = train(en_sen, dec_sen, dec_mask, encoder, decoder, encoder_optimizer, decoder_optimizer)
        
        print_loss_total += loss
        plot_loss_total  += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
   # show_plot(plot_losses)

batch_size = 128
hidden_size = 32
n_layers = 1
input_size = 10
encoder = EncoderRNN(input_size,hidden_size= hidden_size,batch_size= batch_size)
print (encoder)


output_size = 10
attn_decoder1 = AttnDecoderRNN(hidden_size, output_size,
                               1, dropout_p=0.1)
print(attn_decoder1)

#batch generator
batch_gen = batchify(pairs_sorted,batch_size)


if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

train_iters(encoder, attn_decoder1,batch_gen,batch_size,7500, print_every=500)
