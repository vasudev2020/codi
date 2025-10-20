#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:21:51 2024

@author: vnedumpozhimana
"""

import random
# random.seed(0)
import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, bos_token_id, pad_token_id, dim_z, dim_emb=100, dim_h=100, dropout=0.01, nlayers=1, max_len=100, max_nodes_per_batch=128, alg='greedy', initrange=0.1):
        super().__init__()
        
        vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        
        self.max_len = max_len
        self.max_nodes_per_batch = max_nodes_per_batch
        self.alg = alg
        assert alg in ['greedy' , 'sample' , 'top5']
        
        self.embed = nn.Embedding(vocab_size, dim_emb)
        self.proj = nn.Linear(dim_h, vocab_size)
        
        self.G = nn.LSTM(dim_emb, dim_h, nlayers, dropout=dropout if nlayers > 1 else 0)
        self.drop = nn.Dropout(dropout)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)
        self.z2emb = nn.Linear(dim_z, dim_emb)

        
    '''z: (bs,dim), input: (l-1,bs), l < max_len'''
    def decode(self, z, input, hidden=None):
        # print(input.size(),z.size())
        input = self.drop(self.embed(input)) + self.z2emb(z) # (l, bs, tok_dim)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden
    
    '''z: latent representation (bs, dim), max_len: maximum length of the string to be generated'''
    def generate(self, z, max_len=None):
        max_len = self.max_len if max_len is None else max_len
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.bos_token_id) # input: (1, bs) - initialize bos_token_id
        hidden = None
        for l in range(max_len-1):
            logits, hidden = self.decode(z, input, hidden)
            if self.alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif self.alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif self.alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
            sents.append(input)
            
        return torch.cat(sents).transpose(0,1)
    

    '''z: (bs, dim), target: (bs, max_len)'''
    def forward(self, z, target=None, max_len=None,weight=None):
        out={}
        if weight is None:  weight = torch.ones(z.size()[0]).to(next(self.parameters()).device)

        index = list(range(len(z)))
        random.shuffle(index)
        index = torch.LongTensor(sorted(index[:self.max_nodes_per_batch]))

        z = z[index]                                        # (min(bs,max_nodes_per_batch), dim)

        # output = self.generate(z, max_len=max_len)          # (min(bs,max_nodes_per_batch), max_len -1)

        if target is not None:  
            target = target[index]                          # (min(bs,max_nodes_per_batch), max_len)
            weight = weight[index]                          # (min(bs,max_nodes_per_batch), max_len)
            target = target.transpose(0,1).contiguous()     # (max_len, min(bs,max_nodes_per_batch))
            input = target[:-1,:]                           # (max_len-1, min(bs,max_nodes_per_batch))
            target = target[1:,:]                           # (max_len-1, min(bs,max_nodes_per_batch))
            logits,_ = self.decode(z, input)
            # out['loss'] = self.loss_rec(logits, target).mean()
            out['loss'] = torch.mul(self.loss_rec(logits, target), weight).sum() / weight.sum()
            # output[torch.eq(target.transpose(0,1), self.pad_token_id)] = self.pad_token_id
            # out['acc'] = 1*torch.all(torch.eq(target.transpose(0,1), output), dim=1) # (bs)
        # else:
        #     out['output'] = output
        
        return out
    
    def loss_rec(self, logits, targets):  
        # print(logits.size(),targets.size())
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.pad_token_id, reduction='none').view(targets.size())
        return loss.sum(dim=0)
    
    
from collections import Counter


class Vocab(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']
        self.nspecial = 5
        
    @staticmethod
    # def build(sents, path, size):
    def build(words, path, size):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        # words = [w for s in sents for w in s]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w') as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))
                
                
                